# Development

## Goals

FerroGPT is presented as a compact applied-AI backend: a ferroelectrics research corpus, a preprocessing pipeline, retrieval with embeddings, and a Flask API that returns grounded answers. Keep the repository focused on those strengths.

## Architecture

FerroGPT is a retrieval-augmented question-answering backend for ferroelectric memory research. Its source papers are about ferroelectrics, with emphasis on hafnium-oxide ferroelectric materials, FeFETs, and related memory-device behavior. It does not fine-tune or train GPT directly. Instead, it preprocesses a domain corpus, generates embeddings for document sections, retrieves relevant context for each user question, and asks the language model to answer using that context.

Request flow:

```text
Frontend form
    |
    v
POST /process-text
    |
    v
src/app.py validates search_query
    |
    v
src/answer.py embeds the question
    |
    v
Relevant corpus sections are selected
    |
    v
OpenAI completion returns a grounded answer
    |
    v
Flask returns JSON
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `.env` or export it before running scripts:

```bash
export OPENAI_API_KEY=replace-with-your-openai-api-key
```

## Rebuilding The Knowledge Base

The repository keeps the source documents and processed CSV so reviewers can inspect the corpus. Generated embeddings are not committed because they are derived artifacts.

Run:

```bash
python -m src.createCSV
```

This script:

1. Converts PDFs in `trainingDataPdfs/` into `.docx` files when needed.
2. Extracts sections from documents in `trainingDataDocs/`.
3. Writes `trainingData.csv`.
4. Generates `embeddings.json` with OpenAI embeddings.

Key functions in the data pipeline:

- `is_pdf` and `is_doc_or_docx` identify supported input files.
- `preprocess` copies existing document files into `trainingDataDocs/` and converts PDFs when needed.
- `extract_data` turns each document into `(title, heading, content, tokens)` records.
- `count_tokens` measures each section using a GPT-2 tokenizer.
- `reduce_long` keeps long sections within a predictable token budget.
- `compute_doc_embeddings` creates embeddings for every processed section.
- `main` writes `trainingData.csv` and `embeddings.json`.

## Running Locally

```bash
flask --app src.app run
```

Use the API:

```bash
curl -X POST http://127.0.0.1:5000/process-text \
  -F "search_query=What causes wake-up behavior in hafnium oxide ferroelectrics?"
```

## Deployment

The included `Procfile` runs:

```bash
gunicorn src.app:app
```

For a deployed environment, configure:

- `OPENAI_API_KEY`
- `CORS_ORIGINS`
- `AWS_S3_BUCKET`, `EMBEDDINGS_KEY`, and `TRAINING_DATA_KEY` if serving data from S3

If `AWS_S3_BUCKET` is not set, the app expects local `embeddings.json` and `trainingData.csv`.

The app can run with local files or hosted data:

- Local mode expects `trainingData.csv` and `embeddings.json`.
- Hosted mode uses `AWS_S3_BUCKET`, `TRAINING_DATA_KEY`, and `EMBEDDINGS_KEY`.
- Runtime credentials should be configured as environment variables on the deployment platform.

## Configuration Hygiene

- Keep runtime credentials in environment variables.
- Keep local-only files such as `.env`, generated cache folders, and generated embeddings out of git.
- Use `CORS_ORIGINS` to allow only trusted frontend domains.
- Use `FLASK_DEBUG=1` only for local debugging.
- Prefer CSV and JSON for reviewable generated artifacts.

## Future Improvements

- Upgrade from the legacy completion and embedding models to current OpenAI APIs.
- Add tests for request validation, prompt construction, and fallback behavior.
- Improve section extraction for PDFs with more reliable document parsing.
- Add response-level caching or streaming if latency becomes a product concern.
- Add a small frontend demo or hosted Loom walkthrough so reviewers can see the end-to-end workflow quickly.
