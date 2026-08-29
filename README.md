# Vocab Finder

An automated, AI-powered Korean vocabulary extraction pipeline designed to process scanned textbook PDFs (such as Sejong Korean 세종한국어) and output structured, deduplicated JSON datasets to seed vocabulary learning web applications and spaced-repetition (SRS) flashcard databases.

---

## Features

- Chunked PDF Processing: Splits large textbook PDFs into page ranges to optimize throughput and stay within model output limits.
- Multimodal AI Extraction: Leverages Google's gemini-3.6-flash model to process scanned book pages without requiring heavy local CUDA, PyTorch, or system OCR configurations.
- Lemmatization to Dictionary Forms (원형): Automatically converts conjugated verbs and adjectives (e.g., 먹었습니다) into their base dictionary forms (먹다).
- Zero-Setup Directory Handling: Automatically initializes missing assets/ directories and generates template .env.example files.
- Clean Output: Deduplicated, structured JSON objects containing base terms, parts of speech, and concise English definitions.

---

## Tech Stack

- Language: Python 3.10+
- PDF Engine: pypdfium2
- AI Engine: Google GenAI SDK (gemini-3.6-flash)
- Environment Management: python-dotenv

---

## Getting Started

### Prerequisites

- macOS / Linux / Windows
- Python 3.10 or higher
- A free Google Gemini API Key from Google AI Studio

### 1. Installation & Environment Setup

Clone the repository and navigate into the project directory:

```bash
git clone [https://github.com/YangHansen/vocab-finder.git](https://github.com/YangHansen/vocab-finder.git)
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

Create a .env file in the root directory and add your Gemini API key:

`GEMINI_API_KEY="your_actual_api_key_here"`

## Usage

### 1. Place your Korean textbook PDF files inside the `assets/` directory:

`
vocab-finder/
├── assets/
│   ├── Book 1.pdf
│   └── Book 2.pdf
├── .env
├── extractor_engine.py
└── requirements.txt
`
### 2. Run the script

```bash
python extractor_engine.py
```

### 3. Select the target PDF number from the interactive terminal prompt.
### The script generates a JSON file inside `assets/` named after the source PDF.