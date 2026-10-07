# PDF RAG API

A small FastAPI microservice that answers questions about a PDF using retrieval-augmented generation (RAG).

## How it works

1. On startup, the PDF is read with PyMuPDF and split into overlapping chunks (500 characters, 100 overlap).
2. Each chunk is embedded with `sentence-transformers` (`all-MiniLM-L6-v2`) and stored in an in-memory FAISS index.
3. `POST /ask` embeds the question, retrieves the 3 nearest chunks, and sends them with the question to a Llama 3 model (`llama3-8b-8192`) hosted on the Groq API.
4. The generated answer is returned as JSON.

## Tech stack

Python, FastAPI, PyMuPDF, sentence-transformers, FAISS (CPU), Groq API

## Setup

```bash
python3 -m venv rag-venv
source rag-venv/bin/activate
pip install -r requirements.txt
```

Set your Groq API key in `main.py` (`GROQ_API_KEY`, empty by default). Keep your key local and never commit it.

## Run

```bash
uvicorn main:app --reload
```

## API

`POST /ask`

```json
{ "question": "Who is mentioned in the document?" }
```

Response:

```json
{ "question": "...", "answer": "..." }
```

Interactive docs are available at `http://127.0.0.1:8000/docs`.

## Configuration

| Setting | Location | Default |
|---|---|---|
| PDF to index | `PDF_PATH` in `main.py` | `siddique_family.pdf` |
| Chunk size / overlap | `CHUNK_SIZE`, `CHUNK_OVERLAP` | 500 / 100 |
| LLM | `GROQ_MODEL` | `llama3-8b-8192` |

## Limitations and future improvements

- The index is rebuilt on every start and held in memory; there is no persistence or document upload endpoint.
- The API key is configured in code; reading it from an environment variable is the next step.
- No automated tests yet.
