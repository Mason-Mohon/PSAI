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

## 📅 April 25 Update

### ✅ Batch 2: Chunk Uploads to Qdrant Cloud

A major update was completed to expand the PSAI knowledge base by uploading additional content into the Qdrant vector database. This included new and reorganized collections covering commentaries, interviews, columns, and more books. All chunks were embedded using **MiniLM** (`all-MiniLM-L6-v2`) and uploaded client-side in batches via the Qdrant Cloud API.

---

### 🆕 New Collections Created

| Collection Name   | Description |
|-------------------|-------------|
| `commentaries`    | Phyllis Schlafly's audio commentary transcripts (2002–2024) and NET-TV commentaries |
| `columns_chunks`  | Human Events columns and other standalone political commentary pieces |
| `interviews`      | Transcribed interviews of Phyllis Schlafly conducted by Mark DePue |
| `book_chunks`     | Three more books were added to the existing `book_chunks` collection |

---

### 📄 Files Processed and Uploaded

#### 🔹 Commentaries
- All JSON files from `2002.json` to `2024.json` in:
  ```
  /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/chunks/batch2/commentaries/
  ```
- Plus NET-TV commentaries:
  ```
  /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/chunks/batch2/NET-TV.json
  ```

#### 🔹 Columns
- Human Events and similar essays in:
  ```
  /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/chunks/batch2/othercolumns.json
  ```

#### 🔹 Interviews
- Single file containing all interview transcript chunks:
  ```
  /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/chunks/batch2/interview.json
  ```

#### 🔹 Books (added to existing `book_chunks` collection)
- `allegiance.json`  
- `choice_not_echo_2014.json`  
- `how_mass_immigration.json`  
Located in:
  ```
  /Users/mason/Desktop/Technical_Projects/PYTHON_Projects/PSAI/chunks/batch2/
  ```

---

### 🧠 Embedding and Upload Pipeline
- All chunks were embedded locally using `all-MiniLM-L6-v2`
- Metadata was preserved and enriched per chunk (e.g., title, author, interview date)
- Batched upload was handled with `embed_and_upload()` for memory safety
- Collections were created or reused via `ensure_collection()`

---

### 🌐 App Integration

The web app’s backend (`app.py`) uses dynamic collection loading with:
```python
def get_available_collections():
    collections = [c.name for c in qdrant_client.get_collections().collections]
    return collections
```
This means **no manual update** is needed when new collections are added to Qdrant.

The Jinja2 `{% for collection in collections %}` loop ensures the sidebar reflects the current available collections automatically.

---

✅ **Next Steps**
- Evaluate newly uploaded content for accuracy in chunking and metadata alignment
- Add biographical sources and the remaining books to further expand `book_chunks` and `interviews`
- Begin integrating LangChain-based collection selection logic (e.g., political vs biographical)
- Consider fallback integration of Google Search as requested