import json
from fastapi import FastAPI
from rag import build_vectorstore, answer_query

# Create FastAPI app
app = FastAPI(title="RAG API", description="Financial Analysis RAG Pipeline")

# Global vector store
vs = None

@app.on_event("startup")
async def startup_event():
    global vs
    print("Building vector store...")
    vs = build_vectorstore()
    print("Vector store ready!")

@app.get("/")
async def root():
    return {"message": "RAG API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "vector_store_ready": vs is not None}

@app.post("/query")
async def query_documents(query_data: dict):
    """POST - Query the RAG system"""
    query_text = query_data.get("query", "")
    k = query_data.get("k", 5)
    
    if not query_text:
        return {"error": "Query text is required"}
    
    if vs is None:
        return {"error": "Vector store not initialized"}
    
    try:
        result = answer_query(vs, query_text, k=k)
        
        # Transform sources to match exact format
        sources = []
        for source in result.get("sources", []):
            source_path = source.get("source_path", "")
            company = "UNKNOWN"
            year = "UNKNOWN"
            
            if source_path:
                filename = source_path.split('/')[-1].replace('.pdf', '')
                if '-' in filename:
                    parts = filename.split('-')
                    if len(parts) >= 2:
                        company = parts[0].upper()
                        year = parts[1]
            
            sources.append({
                "company": company,
                "year": year,
                "excerpt": source.get("excerpt", ""),
                "page": source.get("page")
            })
        
        return {
            "query": result.get("query", ""),
            "answer": result.get("answer", ""),
            "reasoning": result.get("reasoning", ""),
            "sub_queries": result.get("sub_queries", []),
            "sources": sources
        }
        
    except Exception as e:
        return {"error": f"Error processing query: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)