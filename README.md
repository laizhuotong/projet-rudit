# 📚 EruditAISystem

**EruditAISystem** is an intelligent system for parsing, embedding, indexing, and semantically searching academic articles from Erudit (XML format). It leverages a multilingual SentenceTransformer model and ChromaDB for embedding-based storage and retrieval.

---

## 🚀 Features

| Feature                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `process_xml()`           | Parses XML files, extracts metadata and content, generates embeddings, and stores them in a persistent database. |
| `semantic_search()`       | Performs multilingual semantic search across the document collection.       |
| `get_recommendations()`   | Retrieves semantically similar documents based on a selected document ID.   |
| `show_document_list()`    | Displays a preview list of indexed documents with titles and IDs.           |

---

## 🛠️ Tech Stack

- **Language Model**: [sentence-transformers](https://www.sbert.net/) (paraphrase-multilingual-MiniLM-L12-v2)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **XML Parsing**: `lxml`
- **CLI & UX**: `tqdm`, `os`, `input()`

---

## 📦 Installation

```bash
pip install sentence-transformers chromadb lxml tqdm matplotlib
```
```OR
pip install -r requirements.txt
```

## 📁 Directory Structure
```
project/
├── xml_articles/             # Contains XML article files (nested folders allowed)
│   └── .../
│       └── ERUDITXSDXXX.xml  # Only files matching this pattern are processed
├── chroma_db/                # Persistent ChromaDB storage (auto-generated)
├── main.py                   # Main script with CLI interaction
```
