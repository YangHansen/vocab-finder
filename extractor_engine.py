import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import pypdfium2 as pdfium

# Load environment variables from .env file automatically
load_dotenv()
# Initialize Gemini Client (Requires GEMINI_API_KEY environment variable)
client = genai.Client()

CHUNK_SIZE = 40  # Process 40 pages per API call to ensure output fits within token limits

def split_pdf_to_chunks(pdf_path, chunk_size=CHUNK_SIZE):
    """Splits a PDF into smaller temporary PDF files."""
    src_pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(src_pdf)
    chunk_paths = []

    print(f"📄 Total pages in '{pdf_path.name}': {total_pages}")
    
    for start_idx in range(0, total_pages, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pages)
        
        # Create a new PDF document for this page range
        chunk_pdf = pdfium.PdfDocument.new()
        chunk_pdf.import_pages(src_pdf, list(range(start_idx, end_idx)))
        
        chunk_filename = Path(f"_temp_chunk_{start_idx + 1}_{end_idx}.pdf")
        chunk_pdf.save(chunk_filename)
        chunk_paths.append((chunk_filename, start_idx + 1, end_idx))
        
    return chunk_paths, total_pages

def process_chunk_with_gemini(chunk_path, start_page, end_page):
    """Uploads a PDF chunk to Gemini and extracts vocabulary JSON."""
    print(f"  ☁️ Uploading pages {start_page}–{end_page} to Gemini...")
    uploaded_file = client.files.upload(file=chunk_path)
    
    # Wait briefly for file state to become ACTIVE if processing is needed
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    prompt = """
    Analyze this section of a Korean textbook. Extract all unique vocabulary terms, verbs, and adjectives found across dialogues, exercises, reading passages, and vocabulary lists.

    Return ONLY a valid JSON array where each object has:
    - "korean": base dictionary form (원형, e.g., convert 먹었습니다 to 먹다)
    - "pos": part of speech ("noun", "verb", "adjective")
    - "english": concise English translation

    Strict Rule: Output ONLY raw JSON code. No markdown formatting, no backticks (```), no introductory or concluding text.
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[uploaded_file, prompt]
        )
        raw_text = response.text.strip()
        
        # Strip markdown code blocks if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
            if raw_text.endswith("```"):
                raw_text = raw_text.rsplit("```", 1)[0]
        
        chunk_vocab = json.loads(raw_text.strip())
        return chunk_vocab

    except json.JSONDecodeError:
        print(f"  ⚠️ Warning: Failed to parse JSON for pages {start_page}–{end_page}. Skipping chunk.")
        return []
    finally:
        # Clean up cloud file and local chunk
        client.files.delete(name=uploaded_file.name)
        Path(chunk_path).unlink(missing_ok=True)

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Run: export GEMINI_API_KEY=\"your_key_here\"")
        sys.exit(1)

    assets_dir = Path("assets")
    pdf_files = sorted(list(assets_dir.glob("*.pdf")))
    
    if not pdf_files:
        print("No PDF files found in 'assets/'.")
        sys.exit(1)

    print("====================================")
    print("   CHUNKED KOREAN VOCAB EXTRACTOR   ")
    print("====================================")
    for idx, pdf in enumerate(pdf_files, 1):
        print(f" [{idx}] {pdf.name}")
        
    choice = input("\nSelect a PDF number to process (or press Enter for [1]): ").strip()
    selected_idx = 0 if choice == "" else int(choice) - 1
    
    selected_pdf = pdf_files[selected_idx]
    
    # 1. Split PDF into temporary chunk files
    chunk_info_list, total_pages = split_pdf_to_chunks(selected_pdf)
    
    all_vocab_map = {}
    
    # 2. Process each chunk through Gemini
    for chunk_path, start_page, end_page in chunk_info_list:
        print(f"\nProcessing chunk: Pages {start_page} to {end_page} of {total_pages}...")
        
        vocab_items = process_chunk_with_gemini(chunk_path, start_page, end_page)
        
        # Deduplicate terms using the base Korean word as dictionary key
        for item in vocab_items:
            korean_word = item.get("korean")
            if korean_word and korean_word not in all_vocab_map:
                all_vocab_map[korean_word] = item
                
        print(f"Retained {len(all_vocab_map)} total unique terms so far.")
        time.sleep(2)  # Respect rate limits between calls

    # 3. Save final consolidated output
    combined_vocab_list = list(all_vocab_map.values())
    output_path = assets_dir / f"{selected_pdf.stem}_vocab.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_vocab_list, f, ensure_ascii=False, indent=2)

    print("\n==========================================")
    print(f"SUCCESS! Extracted {len(combined_vocab_list)} total unique vocabulary terms.")
    print(f"Saved to: {output_path}")
    print("==========================================")

if __name__ == "__main__":
    main()