# Vocab Finder

An automated, AI-powered Korean vocabulary extraction pipeline designed to process scanned textbook PDFs (such as Sejong Korean *세종한국어*) and generate structured datasets categorized by **chapter/lesson**, **semantic theme**, and **part of speech**—perfect for seeding vocabulary learning web apps and spaced-repetition (SRS / Anki) flashcard decks.

---

## Key Features

- **Multimodal AI Extraction**: Direct vision processing of scanned textbook pages using Google's Gemini models without requiring local CUDA, PyTorch, or OCR dependencies (Tesseract, PaddleOCR).
- **Chapter & Lesson Categorization**: Automatically detects and attributes vocabulary to textbook chapters and units (e.g., `제1과 약속`, `Lesson 1: Appointments`).
- **Thematic Grouping**: Categorizes terms into semantic themes (e.g., *Food & Dining*, *Travel & Places*, *Emotions*, *Daily Life*).
- **Lemmatization to Dictionary Forms (원형)**: Automatically converts conjugated verbs and adjectives (e.g., 먹었습니다, 갔어요) into base dictionary forms (먹다, 가다).
- **Contextual Example Sentences**: Extracts natural textbook dialogue and reading sentences (`example_korean` and `example_english`) for effective contextual learning.
- **Multiple Output Formats**:
  - **Flat Master List (`*_vocab.json`)**: Deduplicated JSON containing terms with chapter and theme metadata.
  - **Hierarchical Grouping (`*_by_chapter.json`)**: Nested dataset structured by Chapter $\rightarrow$ Theme $\rightarrow$ Vocabulary.
  - **Anki / SRS Flashcards (`*_anki.tsv`)**: Direct 1-click importable tab-separated file for Anki and SRS flashcard apps.
- **Chunked PDF Engine**: Splits large textbook PDFs into small page ranges (default 8 pages) so Gemini extracts exhaustively instead of sampling. Ranges that hit the output token limit are automatically split further.
- **Interactive & CLI Support**: Run with an interactive terminal menu or pass CLI flags for batch automation.
- **Robust Error Handling**: Built-in exponential backoff retries and structured JSON response enforcement.

---

## Tech Stack

- **Language**: Python 3.10+
- **AI Engine**: Google GenAI SDK (`gemini-3.5-flash-lite` by default; configurable)
- **PDF Processing**: `pypdfium2`
- **Configuration**: `python-dotenv`

---

## Getting Started

### Prerequisites

- macOS / Linux / Windows
- Python 3.10 or higher
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 1. Installation & Environment Setup

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/YangHansen/vocab-finder.git
cd vocab-finder
```

Create and activate a Python virtual environment:

```bash
# macOS / Linux
python3 -m venv env
source env/bin/activate

# Windows (Command Prompt)
# venv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory (or copy from `.env.example`) and add your Gemini API key:

```env
GEMINI_API_KEY="your_actual_api_key_here"
# Optional: override model (default: gemini-3.5-flash-lite, ~500 free-tier RPD)
# GEMINI_MODEL="gemini-3.1-flash-lite"
```

---

## Usage

### 1. Place Textbook PDFs in `assets/`

Add your scanned Korean textbook PDF files inside the `assets/` folder:

```text
vocab-finder/
├── assets/
│   ├── Korean_3A.pdf
│   └── Korean_3B.pdf
├── .env
├── extractor_engine.py
├── requirements.txt
└── README.md
```

### 2. Run the Extractor

#### Interactive Mode
Simply run the script without flags to select your PDF from an interactive prompt:

```bash
python extractor_engine.py
```

```text
====================================
   CHUNKED KOREAN VOCAB EXTRACTOR   
====================================
 [1] Korean_3A.pdf
 [2] Korean_3B.pdf
 [3] Process All PDFs

Select a PDF number to process (or press Enter for [1]): 1
```

#### CLI / Batch Mode
You can also automate extraction using command-line arguments:

```bash
# Process a specific PDF
python extractor_engine.py --pdf "Korean_3A.pdf"

# Process all PDFs in assets/ in batch
python extractor_engine.py --all

# Customize page chunk size (default: 8 pages; use smaller values for dense scanned textbooks)
python extractor_engine.py --pdf "Korean_3A.pdf" --chunk-size 8

# Specify a custom Gemini model (3.5-flash-lite is the free-tier default, ~500 RPD)
python extractor_engine.py --pdf "Korean_3A.pdf" --model "gemini-3.1-flash-lite"

# Disable specific export formats
python extractor_engine.py --all --no-anki
```

### CLI Arguments Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--pdf` | `None` | Specific PDF file name or path inside `assets/` |
| `--all` | `False` | Batch process all PDFs inside `assets/` |
| `--chunk-size`, `-c` | `8` | Number of PDF pages per chunk sent to Gemini. Smaller chunks extract more completely from scanned textbooks. |
| `--model`, `-m` | `gemini-3.5-flash-lite` | Gemini model ID. Flash Lite is ~500 RPD on the free tier; regular Flash/Pro are ~20 RPD. |
| `--normalize` | `False` | Rewrite chapter labels in existing `assets/*_vocab.json` files (no API call). |
| `--no-anki` | `False` | Disable generating the `*_anki.tsv` file |
| `--no-grouped` | `False` | Disable generating the `*_by_chapter.json` file |

---

## Output Datasets

For each processed PDF (e.g., `Korean_3A.pdf`), the engine generates three output files in `assets/`:

### 1. Flat Master List (`assets/Korean_3A_vocab.json`)
Deduplicated array of vocabulary items enriched with chapter, theme, and example context:

```json
[
  {
    "korean": "약속",
    "pos": "noun",
    "english": "appointment, promise",
    "chapter": "제1과 약속",
    "theme": "Appointments & Schedules",
    "example_korean": "내일 친구와 약속이 있어서 만나요.",
    "example_english": "I have an appointment with a friend tomorrow, so we are meeting."
  },
  {
    "korean": "늦다",
    "pos": "verb",
    "english": "to be late",
    "chapter": "제1과 약속",
    "theme": "Appointments & Schedules",
    "example_korean": "시간에 늦지 않게 오세요.",
    "example_english": "Please come on time so you are not late."
  }
]
```

### 2. Chapter & Theme Grouped Hierarchy (`assets/Korean_3A_by_chapter.json`)
Hierarchical structure for building unit-based curriculum interfaces:

```json
[
  {
    "chapter": "제1과 약속",
    "total_words": 28,
    "themes": [
      {
        "theme": "Appointments & Schedules",
        "count": 16,
        "vocabulary": [
          {
            "korean": "약속",
            "pos": "noun",
            "english": "appointment, promise",
            "example_korean": "내일 친구와 약속이 있어요.",
            "example_english": "I have an appointment tomorrow."
          }
        ]
      }
    ]
  }
]
```

### 3. Anki / SRS Flashcard Export (`assets/Korean_3A_anki.tsv`)
Tab-separated spreadsheet formatted for direct import into Anki decks:

```text
Korean	English	Part of Speech	Chapter	Theme	Example (Korean)	Example (English)
약속	appointment, promise	noun	제1과 약속	Appointments & Schedules	내일 친구와 약속이 있어요.	I have an appointment tomorrow.
```

---

## License

MIT License. Feel free to use and adapt this pipeline for your Korean language study applications.