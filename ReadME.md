
# PDF to Podcast

> ** Work in Progress** Actively in development. Some features may be incomplete or subject to change.

## About

A FastAPI application that transforms any PDF (via URL) into a fully generated, multi-speaker podcast audio file. Users provide a PDF link and a topic demand, and the app uses Google Gemini to generate a structured transcript and synthesise it into realistic two-voice audio.

## Tech Stack

- **FastAPI** 
- **Google Gemini** (gemini-2.5-flash + TTS) — transcript generation and multi-speaker audio synthesis
- **LangChain + FAISS** — PDF ingestion, chunking, and semantic vector search for context retrieval
- **PDFMiner** PDF text extraction
- **Python** core application logic

## How It Works

1. A PDF URL is validated and loaded
2. The document is chunked and stored in a FAISS vector store
3. The user's demand is analysed and matched against the most relevant chunks
4. Gemini generates a natural two-speaker podcast transcript
5. The transcript is synthesised into audio using Gemini's multi-speaker TTS and exported as WAV/MP3

## Features (In Progress)

- [x] PDF URL validation and loading
- [x] Semantic chunking and vector search with FAISS
- [x] AI-generated podcast transcript via Gemini
- [x] Multi-speaker audio synthesis with prebuilt Gemini voices
- [ ] REST API endpoints (FastAPI integration)
- [ ] Database storage for transcripts
- [ ] Frontend UI
- [ ] Support for local PDF uploads

## Run Locally

```bash
# Clone the repo

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

## Goals & Learning

This project was built to explore AI-powered document processing pipelines, specifically combining retrieval-augmented generation (RAG) with generative audio. 

Key areas of focus include multi-speaker TTS with Gemini, semantic search with FAISS, and building modular, production-ready FastAPI services.