# FerroGPT

FerroGPT is a Flask API for retrieval-augmented question answering over a ferroelectric memory research corpus. The included papers focus on ferroelectrics, especially hafnium-oxide-based ferroelectric materials and memory devices. The app converts those research documents into searchable sections, ranks the most relevant context with OpenAI embeddings, and generates a concise answer grounded in that context.

[Try it out here](https://electrons.ece.gatech.edu/ferrogpt/]

## Highlights

- Flask API with a small `/process-text` endpoint for question answering.
- Retrieval pipeline that embeds ferroelectrics research papers and selects relevant document sections before generation.
- Included sample corpus in `trainingDataDocs/`, `trainingDataPdfs/`, and `trainingData.csv` so reviewers can inspect or rebuild the knowledge base.
- Environment-based configuration for API keys, CORS origins, local data files, and optional S3-hosted data.
- Heroku-compatible `Procfile` and `runtime.txt` for deployment.

## How It Works

1. `src/createCSV.py` processes source documents into `trainingData.csv`.
2. The embedding workflow creates `embeddings.json` from the processed corpus.
3. `src/app.py` exposes `/process-text` for user questions.
4. `src/answer.py` embeds each question, retrieves the closest document sections, builds a grounded prompt, and returns an answer.

## Project Structure

```text
.
├── src/
│   ├── __init__.py         # Python package marker
│   ├── app.py              # Flask routes and request handling
│   ├── answer.py           # Retrieval and answer-generation logic
│   ├── createCSV.py        # Corpus preprocessing and embedding generation
│   └── question.py         # Manual query script
├── trainingDataDocs/       # Source research documents used for the corpus
├── trainingDataPdfs/       # Source PDFs used by the preprocessing script
├── trainingData.csv        # Processed document sections
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn process for deployment
├── runtime.txt             # Python runtime for Heroku-style hosts
└── DEVELOPMENT.md          # Local setup and contributor notes
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create local environment settings:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` or export it in your shell. To run against local data, generate `embeddings.json` once:

```bash
python -m src.createCSV
```

Start the API:

```bash
flask --app src.app run
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:5000/process-text \
  -F "search_query=What improves endurance in HfO2-based FeFETs?"
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for embeddings and completions. |
| `CORS_ORIGINS` | Comma-separated list of allowed browser origins. Leave empty to disable CORS. |
| `AWS_S3_BUCKET` | Optional S3 bucket for production data. If omitted, local files are used. |
| `EMBEDDINGS_KEY` | Local path or S3 key for embeddings. Defaults to `embeddings.json`. |
| `TRAINING_DATA_KEY` | Local path or S3 key for processed sections. Defaults to `trainingData.csv`. |
| `OPENAI_COMPLETIONS_MODEL` | Completion model name. Defaults to `text-davinci-003`. |
| `OPENAI_EMBEDDING_MODEL` | Embedding model name. Defaults to `text-embedding-ada-002`. |
| `MAX_SECTION_LEN` | Token budget for retrieved context. Defaults to `500`. |

## API

`POST /process-text`

Form field:

- `search_query`: the user question.

Example response:

```json
{
  "message": "..."
}
```

## Notes

Runtime credentials live in environment variables, and generated files can be rebuilt locally with the workflow in `DEVELOPMENT.md`.
