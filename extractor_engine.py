import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
import pypdfium2 as pdfium

CHUNK_SIZE = 40  # Process 40 pages per API call to stay within model output limits

def ensure_environment_files():
    """Ensures .env.example exists and checks for GEMINI_API_KEY."""
    env_example_path = Path(".env.example")
    
    # 1. Automatically create .env.example if missing
    if not env_example_path.exists():
        with open(env_example_path, "w", encoding="utf-8") as f:
            f.write("# Gemini API Configuration\n")
            f.write('GEMINI_API_KEY="YourApiKeyHere"\n')
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

def split_pdf_to_chunks(pdf_path, chunk_size=CHUNK_SIZE):
    """Splits a PDF into smaller temporary PDF files."""
    src_pdf = pdfium.PdfDocument(pdf_path)
    total_pages = len(src_pdf)
    chunk_paths = []

    print(f"\nTotal pages in '{pdf_path.name}': {total_pages}")
    
    for start_idx in range(0, total_pages, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pages)
        
        # Create a new PDF document for this page range
        chunk_pdf = pdfium.PdfDocument.new()
        chunk_pdf.import_pages(src_pdf, list(range(start_idx, end_idx)))
        
        chunk_filename = Path(f"_temp_chunk_{start_idx + 1}_{end_idx}.pdf")
        chunk_pdf.save(chunk_filename)
        chunk_paths.append((chunk_filename, start_idx + 1, end_idx))
        
    return chunk_paths, total_pages

def process_chunk_with_gemini(client, chunk_path, start_page, end_page):
    """Uploads a PDF chunk to Gemini and extracts vocabulary JSON."""
    print(f"  Uploading pages {start_page}–{end_page} to Gemini...")
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
        print(f"  Warning: Failed to parse JSON for pages {start_page}–{end_page}. Skipping chunk.")
        return []
    finally:
        # Clean up cloud file and local temporary chunk
        client.files.delete(name=uploaded_file.name)
        Path(chunk_path).unlink(missing_ok=True)

def main():
    # 1. Setup environment and check secrets
    ensure_environment_files()
    
    # 2. Check and prepare assets directory & PDF input
    assets_dir, pdf_files = ensure_assets_directory()

    # 3. Initialize Gemini API Client
    client = genai.Client()

    print("====================================")
    print("   CHUNKED KOREAN VOCAB EXTRACTOR   ")
    print("====================================")
    for idx, pdf in enumerate(pdf_files, 1):
        print(f" [{idx}] {pdf.name}")
        
    choice = input("\nSelect a PDF number to process (or press Enter for [1]): ").strip()
    selected_idx = 0 if choice == "" else int(choice) - 1
    
    if not (0 <= selected_idx < len(pdf_files)):
        print("Invalid selection.")
        sys.exit(1)
        
    selected_pdf = pdf_files[selected_idx]
    
    # 4. Split PDF into temporary chunk files
    chunk_info_list, total_pages = split_pdf_to_chunks(selected_pdf)
    
    all_vocab_map = {}
    
    # 5. Process each chunk through Gemini
    for chunk_path, start_page, end_page in chunk_info_list:
        print(f"\nProcessing chunk: Pages {start_page} to {end_page} of {total_pages}...")
        
        vocab_items = process_chunk_with_gemini(client, chunk_path, start_page, end_page)
        
        # Deduplicate terms using the base Korean word as dictionary key
        for item in vocab_items:
            korean_word = item.get("korean")
            if korean_word and korean_word not in all_vocab_map:
                all_vocab_map[korean_word] = item
                
        print(f"Retained {len(all_vocab_map)} total unique terms so far.")
        time.sleep(2)  # Respect rate limits between calls

    # 6. Save final consolidated output
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