# RAG API

A simple API that answers questions about financial documents using RAG (Retrieval Augmented Generation).

## What it does

- Reads PDF documents from the data folder
- Creates a searchable database of the content
- Answers questions about the documents
- Returns sources and reasoning for each answer

## Setup

1. Install Python packages:

```
pip install -r requirements.txt
```

2. Add your Cohere API key to a .env file:

```
COHERE_API_KEY=your_key_here
```

3. Put your PDF files in the data folder

4. Run the API:

```
python main.py
```

## Usage

The API runs on http://localhost:8000

Send a POST request to /query with:

```json
{
  "query": "What was Microsoft's revenue in 2023?",
  "k": 5
}
```

## Files

- `main.py` - Starts the API server
- `rag.py` - Main RAG logic
- `extractor.py` - Extracts text from PDFs
- `chunking.py` - Splits text into chunks
- `api.py` - API endpoints
