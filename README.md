# Intelligent Doc Search

A portfolio project demonstrating Retrieval-Augmented Generation (RAG) with Python, FastAPI, LangChain, and Docker.

## Features

- Ingests PDF documents
- Splits text into chunks
- Creates embeddings using OpenAI or HuggingFace
- Stores vectors locally in ChromaDB
- Exposes a FastAPI endpoint for question answering
- Supports both OpenAI and HuggingFace LLMs

## Files

- `main.py` - FastAPI application entry point
- `ingest.py` - PDF ingestion and vector store creation script
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container build instructions
- `.env.example` - Sample environment configuration

## Setup

1. Copy environment template:

```bash
cd "c:\Users\Code Farm\Downloads\intelligent doc search"
copy .env.example .env
```

2. Edit `.env` and provide your API key:

```env
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Ingest a PDF

Run the ingestion script to index a PDF into the local Chroma vector store:

```bash
python ingest.py --pdf path\to\document.pdf
```

## Run the API

Start the FastAPI server locally:

```bash
uvicorn main:app --reload
```

## Query the service

Send a POST request to `/query`:

```bash
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What is this document about?\"}"
```

## Docker

Build the container:

```bash
docker build -t intelligent-doc-search .
```

Run the container:

```bash
docker run --rm --env-file .env -p 8000:8000 intelligent-doc-search
```

## Notes

- Ensure the vector store is created before querying the API.
- For HuggingFace mode, update `.env` with `EMBEDDING_PROVIDER=huggingface` and `LLM_PROVIDER=huggingface`.
- This sample is built for clarity and can be expanded with authentication, metadata search, or additional sources.
