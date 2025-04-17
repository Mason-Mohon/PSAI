# Phyllis Schlafly AI (PSAI) – Data Ingestion, Chunking, and Vectorization Summary - update 4/14/2025

## Overview

The PSAI project aims to develop a retrieval-augmented generation (RAG) system powered by Phyllis Schlafly’s written corpus. The ultimate goal is to build an AI assistant that accurately reflects her worldview, writing style, and arguments by embedding and indexing her work across multiple publication types.

This writeup summarizes the current data preparation and vectorization steps, including source locations, chunking logic, metadata structure, and Qdrant vector database setup.

---

## 📁 Data Sources and Structure

The source materials are originally stored under:
D:\Technical_projects\PSAI\


We currently have **three main categories** of source materials:

### 1. **Books** (`chunks/books/`)
- Format: Individual `.json` files per book.
- Files are derived from `.docx` and `.pdf` manuscripts.
- Current books processed:
  - *How the Republican Party Became Pro-Life* (2015)
  - *Who Killed the American Family* (2014)
  - *The Supremacists* (2004)

#### 📄 JSON Schema:
```json
[
  {
    "author": "Phyllis Schlafly",
    "book_title": "Book Title Here",
    "publication_year": 2015,
    "text": "Chunked text here..."
  }
]
```
### 2. Phyllis Schlafly Reports (PSR) (chunks/psr_chunks.json)
Format: Single large JSON file with chunked reports across multiple decades.

Metadata includes title arrays, subjects, date, and source file.

#### 📄 JSON Schema:
```json
[
  {
    "text": "Chunk text here...",
    "metadata": {
      "title": ["Title 1", "Title 2"],
      "date": "August, 1967",
      "author": "Phyllis Schlafly",
      "subjects": ["National Defense", "Latin America"],
      "page_number": 1,
      "source_file": "PSCA_PSR_01_01_196708.pdf",
      "doc_type": "Phyllis Schlafly Report"
    }
  }
]
```
### 3. Phyllis Schlafly Columns (PSC) (chunks/psc_chunks/psc_YYYY_all_chunks.json)
Format: One JSON file per year (1973–2014).

Each file contains year, chunk_count, and chunks list.

📄 JSON Schema:
```json
{
  "year": "1973",
  "chunk_count": 218,
  "chunks": [
    {
      "text": "Chunk text here...",
      "metadata": {
        "title": "Gasoline Rationing",
        "date": "November 23, 1973",
        "author": "Phyllis Schlafly",
        "subjects": [],
        "page_number": 1,
        "source_file": "PSR 1973-11-23.docx",
        "doc_type": "Phyllis Schlafly Column",
        "chunk_id": 1,
        "total_chunks": 5
      }
    }
  ]
}
```

🧱 Chunking Strategy
All documents were chunked using the following principles:

Target size: ~1000 characters per chunk.

Paragraph boundaries preserved where possible.

Metadata is attached to each chunk.

Books are split at paragraph level and embedded with minimal overlap.

All metadata is retained for RAG-enhanced searchability.

🤖 Embedding and Vectorization Pipeline
🧠 Embedding Model
Model: all-MiniLM-L6-v2 via sentence-transformers

Embedding Dimension: 384

Similarity Metric: Cosine

💾 Vector Store
Database: Qdrant Cloud

API Auth: Stored in .env file at D:\Technical_projects\PSAI\.env:

QDRANT_API_KEY

QDRANT_URL

🗂️ Qdrant Collections
Each source is stored in its own collection:

Collection Name	Source Type
book_chunks	Phyllis Schlafly Books
psr_chunks	Phyllis Schlafly Reports
psc_chunks	Phyllis Schlafly Columns
📜 Embedding and Upload Script (Notebook)
The vectorization script is saved as a Jupyter notebook under:

Copy
Edit
embed_ps_chunks_qdrant.ipynb
Key Features:
Loads chunks from all three sources.

Automatically creates Qdrant collections (if not already present).

Embeds each chunk using MiniLM.

Uploads using UUIDs for vector IDs.

Appends to existing collections (supports incremental uploads).

Retains full metadata as Qdrant payload for searchability and filtering.

🔄 Extensibility
The pipeline is fully extensible:

🆕 New books, reports, or columns can be chunked and uploaded with the same format.

✅ You can search each collection independently or create an aggregated interface for unified search.

🛠 The RAG system can use these vectorized embeddings directly via semantic search + retrieval.