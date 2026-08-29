import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import pypdfium2 as pdfium

# Scanned textbooks overflow a single generation if chunks are too large: the model
# returns valid JSON but only a sampled subset.
DEFAULT_CHUNK_SIZE = 8
# Free-tier Flash Lite has ~500 RPD vs 20 RPD on Flash/Pro. Native PDF + image input.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_OUTPUT_TOKENS = 65536
API_CALL_PAUSE_SECONDS = 4
# Abort hung Gemini calls.
REQUEST_TIMEOUT_SECONDS = 240
REQUEST_TIMEOUT_MS = REQUEST_TIMEOUT_SECONDS * 1000
# Stop a run after this many unsuccessful Gemini calls so retries/splits cannot drain RPD.
DEFAULT_MAX_FAILURES = 5

def ensure_environment_files():
    """Ensures .env.example exists and checks for GEMINI_API_KEY."""
    env_example_path = Path(".env.example")
    
    # 1. Automatically create .env.example if missing
    if not env_example_path.exists():
        with open(env_example_path, "w", encoding="utf-8") as f:
            f.write("# Gemini API Configuration\n")
            f.write('GEMINI_API_KEY="YourApiKeyHere"\n')
            f.write('# GEMINI_MODEL="gemini-3.5-flash-lite"\n')
        print("Created '.env.example'. Copy this to '.env' and insert your Gemini API Key.")

    # 2. Load environment variables from .env if present
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\nError: GEMINI_API_KEY environment variable is not set!")
        print("   1. Create a '.env' file in this folder (refer to '.env.example').")
        print("   2. Add your key: GEMINI_API_KEY=\"your_api_key_here\"")
        print("   3. Get a free key at: https://aistudio.google.com/")
        sys.exit(1)
        
    return api_key

def ensure_assets_directory():
    """Ensures assets/ folder exists and checks for PDF files."""
    assets_dir = Path("assets")
    
    # Create assets/ folder if it doesn't exist
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        print("'assets/' folder not found. Created a new 'assets/' directory.")
        print("Please place your Korean textbook PDF file(s) inside 'assets/' and re-run the script.")
        sys.exit(0)
        
    pdf_files = sorted(list(assets_dir.glob("*.pdf")))
    
    if not pdf_files:
        print("No PDF files found inside the 'assets/' folder!")
        print("Please place your textbook PDF(s) inside 'assets/' and re-run the script.")
        sys.exit(0)
        
    return assets_dir, pdf_files

def get_pdf_page_count(pdf_path):
    """Returns the number of pages in a PDF."""
    return len(pdfium.PdfDocument(pdf_path))

def iter_page_ranges(total_pages, chunk_size, start_page=1):
    """Yields inclusive 1-based (start, end) page ranges, optionally skipping completed pages."""
    start_idx = max(0, start_page - 1)
    for start_idx in range(start_idx, total_pages, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pages)
        yield start_idx + 1, end_idx

def write_pdf_chunk(pdf_path, start_page, end_page):
    """Writes a temporary PDF covering the inclusive 1-based page range."""
    src_pdf = pdfium.PdfDocument(pdf_path)
    chunk_pdf = pdfium.PdfDocument.new()
    chunk_pdf.import_pages(src_pdf, list(range(start_page - 1, end_page)))
    chunk_filename = Path(f"_temp_chunk_{start_page}_{end_page}.pdf")
    chunk_pdf.save(chunk_filename)
    return chunk_filename

def _finish_reason_name(response):
    """Returns the candidate finish_reason as a string, or None."""
    if not response.candidates:
        return None
    reason = response.candidates[0].finish_reason
    if reason is None:
        return None
    return reason.name if hasattr(reason, "name") else str(reason)

def _normalize_vocab_payload(payload):
    """Accepts a raw JSON array or a wrapped {vocabulary: [...]} object."""
    if isinstance(payload, dict) and "vocabulary" in payload:
        payload = payload["vocabulary"]
    return payload if isinstance(payload, list) else []

class DailyQuotaExceeded(Exception):
    """Raised when the model's free-tier requests-per-day (RPD) quota is exhausted."""

    def __init__(self, model, original):
        self.model = model
        self.original = original
        super().__init__(str(original))

class ExtractionBudgetExceeded(Exception):
    """Raised when too many Gemini calls failed in one run (protects remaining RPD)."""

    def __init__(self, wasted, limit):
        self.wasted = wasted
        self.limit = limit
        super().__init__(
            f"Stopped after {wasted} failed API requests (limit {limit}) to protect daily quota."
        )

class RequestBudget:
    """Counts unsuccessful generate_content calls and halts when the cap is reached."""

    def __init__(self, max_failures=DEFAULT_MAX_FAILURES):
        self.max_failures = max_failures
        self.wasted = 0

    def note_waste(self, reason=""):
        self.wasted += 1
        cap = "unlimited" if self.max_failures <= 0 else str(self.max_failures)
        suffix = f" ({reason})" if reason else ""
        print(f"  Failed API calls this run: {self.wasted}/{cap}{suffix}")
        if self.max_failures > 0 and self.wasted >= self.max_failures:
            raise ExtractionBudgetExceeded(self.wasted, self.max_failures)

def _is_rate_limit_error(exc):
    """True for free-tier RPM/RPD quota errors."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in (429, "429"):
        return True
    text = str(exc).lower()
    return any(token in text for token in ("429", "resource_exhausted", "rate limit", "quota"))

def _is_daily_quota_error(exc):
    """True when the 429 is the daily RPD cap, which waiting cannot recover."""
    if not _is_rate_limit_error(exc):
        return False
    text = str(exc).lower()
    daily_markers = ("per_day", "per day", "rpd", "daily quota", "requests per day")
    if any(marker in text for marker in daily_markers):
        return True
    # RPM errors usually include a retry delay. A bare quota 429 is the daily cap.
    return "retry in" not in text and "retry-after" not in text

def _is_timeout_error(exc):
    """True when the HTTP client gave up waiting for Gemini."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = ("timeout", "timed out", "deadline exceeded", "readtimeout", "connecttimeout")
    return any(marker in name or marker in text for marker in markers)

def _build_generate_config(model):
    """Build generation config, using Gemini 3-only options only on 3.x models."""
    kwargs = {
        "response_mime_type": "application/json",
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "http_options": types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    }
    name = (model or "").lower()
    if "gemini-3" in name:
        kwargs["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_HIGH
        kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL,
        )
    elif "gemini-2.5" in name:
        # Disable thinking so free-tier output quota goes to vocabulary JSON.
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    return types.GenerateContentConfig(**kwargs)

def process_chunk_with_gemini(client, chunk_path, start_page, end_page, model=DEFAULT_MODEL, max_retries=3, budget=None):
    """Uploads a PDF chunk to Gemini and extracts structured vocabulary JSON with retries.

    Returns (vocab_items, finish_reason, parse_ok). parse_ok is False when JSON could
    not be decoded; finish_reason is MAX_TOKENS when generation hit the output cap.
    """
    print(f"  Uploading pages {start_page}–{end_page} to Gemini...")

    uploaded_file = None
    finish_reason = None
    try:
        uploaded_file = client.files.upload(file=chunk_path)

        waited = 0
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            waited += 2
            if waited >= REQUEST_TIMEOUT_SECONDS:
                print(f"  File processing timed out after {REQUEST_TIMEOUT_SECONDS}s for pages {start_page}–{end_page}.")
                if budget:
                    budget.note_waste("upload processing timeout")
                return [], "TIMEOUT", False
            uploaded_file = client.files.get(name=uploaded_file.name)

        prompt = """
        Analyze this section of a Korean textbook. Extract EVERY unique vocabulary term,
        verb, adjective, adverb, particle, and expression found across dialogues,
        exercises, reading passages, and vocabulary lists. Do not summarize, sample,
        or skip later pages in this section. If a printed vocabulary list appears,
        transcribe it completely.

        Chapter labels (use one consistent form for the whole lesson):
        - Some books insert 1–2 divider pages before a new lesson. Those pages show a large
          lesson mark such as "2과" plus the Korean title. They are NOT identified by color;
          the design varies. Vocabulary after a divider belongs to that lesson until the next divider.
        - Other books (including some Practical vocab books) have no divider pages. Then read
          the lesson number/title from running headers, "제N과", or vocabulary-list headings.
        - Part 01 / Part 02 / 어휘 / 문법 inside a lesson are NOT new chapters.
        - Always set "chapter" to "N과 {Korean title}" (e.g. "2과 어제 친구를 우연히 기차역에서 만났어요").
          If the title is not visible, use "N과". Never use "2-1", "02", "2과-1", or English-only labels.

        For each vocabulary entry, provide:
        - "korean": base dictionary form (원형, e.g., convert 먹었습니다 to 먹다, 갔어요 to 가다)
        - "pos": part of speech ("noun", "verb", "adjective", "adverb", "expression", "particle")
        - "english": concise English translation / definition
        - "chapter": as specified above, or "General" if no lesson number is visible
        - "theme": the semantic theme or category of the vocabulary (e.g., "Appointments & Schedules", "Food & Dining", "Travel & Transportation", "Health & Body", "Emotions & Personality", "Daily Life", "Work & Study")
        - "example_korean": a contextual example sentence from the textbook if available (or empty string if none)
        - "example_english": English translation of the example sentence if available (or empty string if none)

        Return ONLY a valid JSON array of objects matching this specification.
        """

        config = _build_generate_config(model)

        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[uploaded_file, prompt],
                    config=config
                )
                finish_reason = _finish_reason_name(response)
                usage = response.usage_metadata
                if usage:
                    print(
                        f"  Model usage: prompt={usage.prompt_token_count} "
                        f"output={usage.candidates_token_count} "
                        f"thoughts={usage.thoughts_token_count} "
                        f"finish={finish_reason}"
                    )

                raw_text = (response.text or "").strip()
                if not raw_text:
                    print(f"  Warning: Empty model response for pages {start_page}–{end_page} ({finish_reason}).")
                    if budget:
                        budget.note_waste("empty response")
                    if finish_reason == "MAX_TOKENS":
                        return [], finish_reason, False
                    if attempt < max_retries:
                        time.sleep(2 * attempt)
                        continue
                    return [], finish_reason, False

                if raw_text.startswith("```"):
                    raw_text = raw_text.split("\n", 1)[1]
                    if raw_text.endswith("```"):
                        raw_text = raw_text.rsplit("```", 1)[0]

                chunk_vocab = _normalize_vocab_payload(json.loads(raw_text.strip()))
                parse_ok = True
                if finish_reason == "MAX_TOKENS":
                    print(
                        f"  Warning: Output truncated at token limit for pages {start_page}–{end_page} "
                        f"({len(chunk_vocab)} terms parsed)."
                    )
                return chunk_vocab, finish_reason, parse_ok

            except json.JSONDecodeError:
                print(f"  Warning: JSON decode attempt {attempt}/{max_retries} failed for pages {start_page}–{end_page} (finish={finish_reason}).")
                if budget:
                    budget.note_waste("invalid JSON")
                # Truncated JSON will not become valid by retrying the same range.
                if finish_reason == "MAX_TOKENS":
                    return [], finish_reason, False
                if attempt < max_retries:
                    time.sleep(2 * attempt)
                else:
                    return [], finish_reason, False
            except (DailyQuotaExceeded, ExtractionBudgetExceeded, KeyboardInterrupt):
                raise
            except Exception as e:
                print(f"  Warning: API call attempt {attempt}/{max_retries} encountered error: {e}")
                if _is_daily_quota_error(e):
                    raise DailyQuotaExceeded(model, e) from e
                if _is_timeout_error(e):
                    print(
                        f"  Request timed out after {REQUEST_TIMEOUT_SECONDS}s "
                        f"for pages {start_page}–{end_page}."
                    )
                    if budget:
                        budget.note_waste("request timeout")
                    return [], "TIMEOUT", False
                if budget:
                    budget.note_waste("api error")
                if attempt < max_retries:
                    if _is_rate_limit_error(e):
                        wait_time = 30 * attempt
                        print(f"  Per-minute rate limit. Waiting {wait_time}s...")
                    else:
                        wait_time = 2 ** attempt
                        print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Failed after {max_retries} attempts for pages {start_page}–{end_page}. Skipping chunk.")
                    return [], finish_reason, False

        return [], finish_reason, False

    finally:
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        Path(chunk_path).unlink(missing_ok=True)

def _should_resplit(page_count, items, finish_reason, parse_ok):
    """True when this range should be split and re-extracted."""
    if page_count <= 1:
        return False
    if not parse_ok or finish_reason in {"MAX_TOKENS", "TIMEOUT"}:
        return True
    # Model often "completes" a large scanned range with a sampled subset (valid JSON).
    # Default 8-page chunks skip this; larger --chunk-size values still get bisected.
    min_expected = page_count * 3
    return page_count > DEFAULT_CHUNK_SIZE and len(items) < min_expected

def extract_vocab_from_range(client, pdf_path, start_page, end_page, model=DEFAULT_MODEL, max_retries=3, budget=None):
    """Extracts vocabulary for a page range, bisecting if output is truncated or unparseable."""
    page_count = end_page - start_page + 1
    chunk_path = write_pdf_chunk(pdf_path, start_page, end_page)
    items, finish_reason, parse_ok = process_chunk_with_gemini(
        client, chunk_path, start_page, end_page, model=model, max_retries=max_retries, budget=budget
    )
    time.sleep(API_CALL_PAUSE_SECONDS)

    if _should_resplit(page_count, items, finish_reason, parse_ok):
        mid = (start_page + end_page) // 2
        print(
            f"  Incomplete extraction for pages {start_page}–{end_page} "
            f"(finish={finish_reason}, parsed={len(items)}). Splitting into "
            f"{start_page}–{mid} and {mid + 1}–{end_page}..."
        )
        left = extract_vocab_from_range(
            client, pdf_path, start_page, mid, model=model, max_retries=max_retries, budget=budget
        )
        right = extract_vocab_from_range(
            client, pdf_path, mid + 1, end_page, model=model, max_retries=max_retries, budget=budget
        )
        return left + right

    print(f"  Extracted {len(items)} terms from pages {start_page}–{end_page}.")
    return items

_HANGUL_RE = re.compile(r"[가-힣]")
_CHAPTER_NUM_RE = re.compile(
    r"""
    ^\s*
    (?:제)?                          # optional 제
    0*(?P<num>\d+)                   # lesson number
    (?:\s*과)?                       # optional 과
    (?:\s*[-–./]\s*\d+)?             # optional part suffix: 1-1, 09-2
    \s*
    (?P<title>.*?)
    \s*$
    """,
    re.VERBOSE,
)
_LESSON_EN_RE = re.compile(
    r"^\s*(?:lesson|unit|chapter)\s*0*(\d+)\s*[:\-–]?\s*(.*)$",
    re.IGNORECASE,
)

def parse_chapter(raw):
    """Returns (lesson_number or None, title_text) from a messy chapter label."""
    text = (raw or "").strip()
    if not text or text.lower() in {"general", "none", "n/a", "unknown"}:
        return None, ""

    match = _CHAPTER_NUM_RE.match(text)
    if match:
        num = int(match.group("num"))
        title = (match.group("title") or "").strip(" -:–|/")
        if title in {"과", "과."}:
            title = ""
        if title.startswith("과 "):
            title = title[2:].strip()
        return num, title

    match = _LESSON_EN_RE.match(text)
    if match:
        return int(match.group(1)), (match.group(2) or "").strip(" -:–|/")

    return None, text

def format_chapter(num, title=""):
    """Canonical chapter label: '3과' or '3과 {title}'."""
    title = (title or "").strip()
    if num is None:
        return title or "General"
    return f"{num}과 {title}".strip() if title else f"{num}과"

def _title_score(title):
    hangul = len(_HANGUL_RE.findall(title or ""))
    return (hangul, len(title or ""))

def harvest_chapter_titles(vocab_lists):
    """Picks the best Korean lesson title seen for each lesson number."""
    best = {}
    for vocab_list in vocab_lists:
        for item in vocab_list:
            num, title = parse_chapter(item.get("chapter"))
            if num is None or not title or _title_score(title)[0] == 0:
                continue
            if num not in best or _title_score(title) > _title_score(best[num]):
                best[num] = title
    return best

def canonicalize_chapters(vocab_list, title_map=None):
    """Rewrites item['chapter'] to a single 'N과 {title}' form per lesson."""
    titles = dict(title_map or {})
    harvested = harvest_chapter_titles([vocab_list])
    for num, title in harvested.items():
        if num not in titles or _title_score(title) > _title_score(titles.get(num, "")):
            titles[num] = title

    for item in vocab_list:
        num, _local_title = parse_chapter(item.get("chapter"))
        if num is None:
            item["chapter"] = (item.get("chapter") or "").strip() or "General"
        else:
            item["chapter"] = format_chapter(num, titles.get(num, ""))
    return vocab_list

def group_vocab_by_chapter(vocab_list):
    """Groups flat vocabulary list into hierarchical chapter -> theme -> words structure."""
    chapters_map = {}
    for item in vocab_list:
        chap = item.get("chapter") or "General"
        theme = item.get("theme") or "General"
        
        if chap not in chapters_map:
            chapters_map[chap] = {}
        if theme not in chapters_map[chap]:
            chapters_map[chap][theme] = []
            
        chapters_map[chap][theme].append({
            "korean": item.get("korean"),
            "pos": item.get("pos", "noun"),
            "english": item.get("english", ""),
            "example_korean": item.get("example_korean", ""),
            "example_english": item.get("example_english", "")
        })
    
    structured_chapters = []
    for chap_title, themes in chapters_map.items():
        theme_list = []
        for theme_title, words in themes.items():
            theme_list.append({
                "theme": theme_title,
                "count": len(words),
                "vocabulary": words
            })
        structured_chapters.append({
            "chapter": chap_title,
            "total_words": sum(t["count"] for t in theme_list),
            "themes": theme_list
        })

    def sort_key(entry):
        num, _title = parse_chapter(entry["chapter"])
        return (num is None, num if num is not None else 10**9, entry["chapter"])

    structured_chapters.sort(key=sort_key)
    return structured_chapters

def export_anki_tsv(vocab_list, output_path):
    """Exports vocabulary to an Anki/SRS-compatible Tab-Separated Values (TSV) file."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Korean", "English", "Part of Speech", "Chapter", "Theme", "Example (Korean)", "Example (English)"])
        for item in vocab_list:
            writer.writerow([
                item.get("korean", ""),
                item.get("english", ""),
                item.get("pos", ""),
                item.get("chapter", ""),
                item.get("theme", ""),
                item.get("example_korean", ""),
                item.get("example_english", "")
            ])

def _checkpoint_path(assets_dir, pdf_path):
    return assets_dir / f"{pdf_path.stem}_checkpoint.json"

def _pdf_fingerprint(pdf_path):
    stat = pdf_path.stat()
    return {"name": pdf_path.name, "size": stat.st_size, "mtime": int(stat.st_mtime)}

def _write_json_atomic(path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)

def load_checkpoint(assets_dir, pdf_path):
    """Returns a valid checkpoint dict, or None if missing/mismatched."""
    path = _checkpoint_path(assets_dir, pdf_path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("pdf") != _pdf_fingerprint(pdf_path):
        print(f"Ignoring checkpoint {path.name} (PDF file changed).")
        return None
    return data

def save_checkpoint(assets_dir, pdf_path, total_pages, last_completed_end_page, vocab_map, chunk_size, model):
    """Persists in-progress vocabulary so a later run can skip completed pages."""
    payload = {
        "pdf": _pdf_fingerprint(pdf_path),
        "total_pages": total_pages,
        "last_completed_end_page": last_completed_end_page,
        "chunk_size": chunk_size,
        "model": model,
        "vocab": list(vocab_map.values()),
    }
    path = _checkpoint_path(assets_dir, pdf_path)
    _write_json_atomic(path, payload)
    print(
        f"  Checkpoint saved: {path.name} "
        f"(through page {last_completed_end_page}/{total_pages}, {len(vocab_map)} terms)."
    )

def clear_checkpoint(assets_dir, pdf_path):
    _checkpoint_path(assets_dir, pdf_path).unlink(missing_ok=True)

def merge_vocab_item(all_vocab_map, item):
    korean_word = item.get("korean")
    if not korean_word:
        return
    if korean_word not in all_vocab_map:
        all_vocab_map[korean_word] = item
        return
    existing = all_vocab_map[korean_word]
    if not existing.get("example_korean") and item.get("example_korean"):
        existing["example_korean"] = item.get("example_korean")
        existing["example_english"] = item.get("example_english", "")

def process_pdf(selected_pdf, client, assets_dir, chunk_size=DEFAULT_CHUNK_SIZE, model=DEFAULT_MODEL, export_anki=True, export_grouped=True, fresh=False, max_failures=DEFAULT_MAX_FAILURES):
    """Processes a single PDF file through the extraction and formatting pipeline."""
    total_pages = get_pdf_page_count(selected_pdf)
    print(f"\nTotal pages in '{selected_pdf.name}': {total_pages} (chunk size: {chunk_size})")

    all_vocab_map = {}
    resume_from_page = 1
    last_completed_end_page = 0

    if not fresh:
        checkpoint = load_checkpoint(assets_dir, selected_pdf)
        if checkpoint:
            for item in checkpoint.get("vocab") or []:
                merge_vocab_item(all_vocab_map, item)
            last_completed_end_page = int(checkpoint.get("last_completed_end_page") or 0)
            resume_from_page = last_completed_end_page + 1
            print(
                f"Resuming from page {resume_from_page} "
                f"({len(all_vocab_map)} terms already saved). Use --fresh to start over."
            )

    budget = RequestBudget(max_failures)

    try:
        for start_page, end_page in iter_page_ranges(total_pages, chunk_size, start_page=resume_from_page):
            print(f"\nProcessing chunk: Pages {start_page} to {end_page} of {total_pages}...")

            vocab_items = extract_vocab_from_range(
                client, selected_pdf, start_page, end_page, model=model, budget=budget
            )

            for item in vocab_items:
                merge_vocab_item(all_vocab_map, item)

            last_completed_end_page = end_page
            print(f"Retained {len(all_vocab_map)} total unique terms so far.")
            save_checkpoint(
                assets_dir, selected_pdf, total_pages, last_completed_end_page,
                all_vocab_map, chunk_size, model,
            )
            _write_vocab_outputs(
                selected_pdf, assets_dir, list(all_vocab_map.values()),
                export_anki, export_grouped, quiet=True,
            )
    except DailyQuotaExceeded:
        if all_vocab_map:
            print(f"\nDaily quota hit. Saving {len(all_vocab_map)} terms extracted so far...")
            save_checkpoint(
                assets_dir, selected_pdf, total_pages, last_completed_end_page,
                all_vocab_map, chunk_size, model,
            )
            _write_vocab_outputs(selected_pdf, assets_dir, list(all_vocab_map.values()), export_anki, export_grouped)
        raise
    except ExtractionBudgetExceeded:
        print(f"\nToo many failed API calls. Saving checkpoint to protect remaining daily quota...")
        save_checkpoint(
            assets_dir, selected_pdf, total_pages, last_completed_end_page,
            all_vocab_map, chunk_size, model,
        )
        if all_vocab_map:
            _write_vocab_outputs(selected_pdf, assets_dir, list(all_vocab_map.values()), export_anki, export_grouped)
        raise
    except KeyboardInterrupt:
        if all_vocab_map or last_completed_end_page:
            print("\nInterrupted. Saving checkpoint so you can continue later...")
            save_checkpoint(
                assets_dir, selected_pdf, total_pages, last_completed_end_page,
                all_vocab_map, chunk_size, model,
            )
            if all_vocab_map:
                _write_vocab_outputs(
                    selected_pdf, assets_dir, list(all_vocab_map.values()),
                    export_anki, export_grouped,
                )
        raise

    combined_vocab_list = list(all_vocab_map.values())
    _write_vocab_outputs(selected_pdf, assets_dir, combined_vocab_list, export_anki, export_grouped)
    clear_checkpoint(assets_dir, selected_pdf)
    print("\n==========================================")
    print(f"SUCCESS! Extracted {len(combined_vocab_list)} total unique terms from {selected_pdf.name}.")
    print("==========================================")

def _write_vocab_outputs(selected_pdf, assets_dir, combined_vocab_list, export_anki, export_grouped, title_map=None, quiet=False):
    """Writes flat JSON, optional grouped JSON, and optional Anki TSV."""
    combined_vocab_list = canonicalize_chapters(combined_vocab_list, title_map=title_map)
    flat_output_path = assets_dir / f"{selected_pdf.stem}_vocab.json"
    with open(flat_output_path, "w", encoding="utf-8") as f:
        json.dump(combined_vocab_list, f, ensure_ascii=False, indent=2)
    if not quiet:
        print(f"\nSaved flat vocabulary list to: {flat_output_path}")

    if export_grouped:
        grouped_data = group_vocab_by_chapter(combined_vocab_list)
        grouped_output_path = assets_dir / f"{selected_pdf.stem}_by_chapter.json"
        with open(grouped_output_path, "w", encoding="utf-8") as f:
            json.dump(grouped_data, f, ensure_ascii=False, indent=2)
        if not quiet:
            print(f"Saved chapter/theme grouped dataset to: {grouped_output_path}")

    if export_anki:
        anki_output_path = assets_dir / f"{selected_pdf.stem}_anki.tsv"
        export_anki_tsv(combined_vocab_list, anki_output_path)
        if not quiet:
            print(f"Saved Anki SRS flashcards export to: {anki_output_path}")

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="AI-powered Korean textbook vocabulary extraction pipeline.")
    parser.add_argument("--pdf", type=str, help="Specific PDF file name or path inside assets/ to process.")
    parser.add_argument("--all", action="store_true", help="Process all PDF files found in assets/ directory.")
    parser.add_argument("--chunk-size", "-c", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Pages per API chunk (default: {DEFAULT_CHUNK_SIZE}; smaller is more complete for scanned textbooks).")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help=f"Gemini model ID to use (default: {DEFAULT_MODEL}).")
    parser.add_argument("--no-anki", action="store_true", help="Disable generating Anki TSV export file.")
    parser.add_argument("--no-grouped", action="store_true", help="Disable generating chapter/theme grouped JSON file.")
    parser.add_argument("--normalize", action="store_true", help="Rewrite chapter labels in existing assets/*_vocab.json files without calling the API.")
    parser.add_argument("--fresh", action="store_true", help="Ignore any checkpoint and extract the PDF from page 1.")
    parser.add_argument("--max-failures", type=int, default=DEFAULT_MAX_FAILURES, help=f"Stop the run after this many failed Gemini calls to protect RPD (default: {DEFAULT_MAX_FAILURES}; 0 = unlimited).")
    return parser.parse_args()

def normalize_existing_outputs(assets_dir, export_anki=True, export_grouped=True):
    """Normalizes chapter labels in existing vocab JSON files and regenerates exports."""
    vocab_paths = sorted(assets_dir.glob("*_vocab.json"))
    loaded = []
    for path in vocab_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not data or "chapter" not in data[0]:
            print(f"Skipping {path.name} (no chapter field).")
            continue
        loaded.append((path, data))

    if not loaded:
        print("No chapter-tagged *_vocab.json files found to normalize.")
        return

    shared_titles = harvest_chapter_titles([data for _path, data in loaded])
    print(f"Harvested titles for {len(shared_titles)} lesson(s):")
    for num in sorted(shared_titles):
        print(f"  {format_chapter(num, shared_titles[num])}")

    for path, data in loaded:
        stem = path.name[: -len("_vocab.json")]
        selected = type("PdfStub", (), {"stem": stem, "name": f"{stem}.pdf"})()
        before = len({item.get("chapter") for item in data})
        canonicalize_chapters(data, title_map=shared_titles)
        after = len({item.get("chapter") for item in data})
        _write_vocab_outputs(selected, assets_dir, data, export_anki, export_grouped, title_map=shared_titles)
        print(f"Normalized {path.name}: {before} chapter labels → {after} lessons.")

def main():
    args = parse_arguments()

    if args.normalize:
        assets_dir = Path("assets")
        if not assets_dir.exists():
            print("No 'assets/' folder found.")
            sys.exit(1)
        normalize_existing_outputs(
            assets_dir,
            export_anki=not args.no_anki,
            export_grouped=not args.no_grouped,
        )
        return

    # 1. Setup environment and check secrets
    ensure_environment_files()
    
    # 2. Check and prepare assets directory & PDF input
    assets_dir, pdf_files = ensure_assets_directory()

    # 3. Initialize Gemini API Client
    client = genai.Client(http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

    export_anki = not args.no_anki
    export_grouped = not args.no_grouped

    # 4. Determine which PDFs to process
    target_pdfs = []
    if args.all:
        target_pdfs = pdf_files
    elif args.pdf:
        matching = [p for p in pdf_files if p.name.lower() == args.pdf.lower() or p.stem.lower() == args.pdf.lower()]
        if not matching:
            print(f"Error: Specified PDF '{args.pdf}' not found in 'assets/' folder.")
            print(f"Available PDFs: {[p.name for p in pdf_files]}")
            sys.exit(1)
        target_pdfs = [matching[0]]
    else:
        # Interactive selection prompt
        print("====================================")
        print("   CHUNKED KOREAN VOCAB EXTRACTOR   ")
        print("====================================")
        for idx, pdf in enumerate(pdf_files, 1):
            print(f" [{idx}] {pdf.name}")
        print(f" [{len(pdf_files) + 1}] Process All PDFs")
            
        choice = input(f"\nSelect a PDF number to process (or press Enter for [1]): ").strip()
        if choice == "":
            selected_idx = 0
            target_pdfs = [pdf_files[0]]
        else:
            try:
                selected_num = int(choice)
                if selected_num == len(pdf_files) + 1:
                    target_pdfs = pdf_files
                elif 1 <= selected_num <= len(pdf_files):
                    target_pdfs = [pdf_files[selected_num - 1]]
                else:
                    print("Invalid selection.")
                    sys.exit(1)
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                sys.exit(1)

    # 5. Process target PDF(s)
    for pdf in target_pdfs:
        print(f"\n==========================================")
        print(f"Starting extraction for: {pdf.name}")
        print("==========================================")
        try:
            process_pdf(
                selected_pdf=pdf,
                client=client,
                assets_dir=assets_dir,
                chunk_size=args.chunk_size,
                model=args.model,
                export_anki=export_anki,
                export_grouped=export_grouped,
                fresh=args.fresh,
                max_failures=args.max_failures,
            )
        except KeyboardInterrupt:
            print("\nStopped. Re-run the same command to continue from the last finished chunk.")
            sys.exit(130)
        except ExtractionBudgetExceeded as e:
            print("\n==========================================")
            print(f"STOPPED: {e.wasted} failed Gemini requests (limit {e.limit}) to protect remaining RPD.")
            print("A checkpoint was saved. Re-run the same command later to continue from the last finished chunk.")
            print("==========================================")
            sys.exit(1)
        except DailyQuotaExceeded as e:
            print("\n==========================================")
            print(f"STOPPED: daily request quota (RPD) exhausted for model '{e.model}'.")
            print("RPD is per model and resets at midnight Pacific Time. Waiting/retrying will not help.")
            print("A checkpoint was saved. Re-run the same command tomorrow (or with another model) to continue.")
            print("Switch to a different model (separate daily pool), for example:")
            print(f'  python extractor_engine.py --pdf "{pdf.name}" --model gemini-3.5-flash-lite')
            print("==========================================")
            sys.exit(1)

if __name__ == "__main__":
    main()
