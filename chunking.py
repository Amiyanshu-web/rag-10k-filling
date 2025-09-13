from typing import List, Dict, Any, Optional
import os

from langchain_experimental.text_splitter import SemanticChunker

from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain.schema.document import Document


def get_semantic_splitter() -> SemanticChunker:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
        buffer_size=2,
    )


def chunk_extracted_content(
    extracted_data: List[Dict[str, Any]], 
    cohere_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    splitter = get_semantic_splitter()
    chunked_results = []
    
    for data in extracted_data:
        if data.get("engine") == "error":
            chunked_results.append(data)
            continue
            
        text = data.get("text", "")
        tables = data.get("tables", [])
        
        text_chunks = []
        if text:
            docs = splitter.create_documents([text])
            text_chunks = [
                {
                    "content": doc.page_content,
                    "type": "text",
                    "chunk_index": idx,
                }
                for idx, doc in enumerate(docs)
            ]
        
        table_chunks = []
        for table_idx, table_content in enumerate(tables):
            table_chunks.append({
                "content": table_content,
                "type": "table",
                "chunk_index": f"table_{table_idx}",
            })
        
        all_chunks = text_chunks + table_chunks
        
        chunked_result = {
            "id": data.get("id"),
            "source_path": data.get("source_path"),
            "engine": data.get("engine"),
            "chunks": all_chunks
        }
        
        chunked_results.append(chunked_result)
    
    return chunked_results


def create_documents_with_metadata(
    extracted_data: List[Dict[str, Any]], 
    cohere_api_key: Optional[str] = None
) -> List[Document]:
	chunked_data = chunk_extracted_content(extracted_data)	
	documents = []

	for doc_data in chunked_data:
		if doc_data.get("engine") == "error":
			continue
			
		for chunk in doc_data.get("chunks", []):
			metadata = {
				"source_id": chunk.get("chunk_index"),
				"source_path": doc_data.get("source_path"),
				"chunk_type": chunk["type"],
				"chunk_index": chunk["chunk_index"],
			}
			
			documents.append(Document(
				page_content=chunk["content"],
				metadata=metadata
			))

	return documents