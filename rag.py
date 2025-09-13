import os
import json
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS  
from langchain_cohere import ChatCohere  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document

from extractor import run as extract_texts
from chunking import create_documents_with_metadata
from dotenv import load_dotenv

load_dotenv()


def _chat() -> ChatCohere:
	return ChatCohere(model="command-r-plus", cohere_api_key=os.getenv("COHERE_API_KEY"))


def build_vectorstore() -> FAISS:
	# Get extracted data from JSONL files
	extracted_data = []
	for file in os.listdir("out"):
		if file.endswith(".jsonl"):
			with open(f"out/{file}", "rb") as f:
				import orjson
				line = f.readline()
				if line:
					extracted_data.append(orjson.loads(line))
	
	# Create documents with metadata
	documents = create_documents_with_metadata(extracted_data, os.getenv("COHERE_API_KEY"))
	
	# Build vector store
	embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
	vs = FAISS.from_documents(documents, embeddings)
	return vs


def retrieve(vs: FAISS, query: str, k: int = 5) -> List[Document]:
	retriever = vs.as_retriever(search_kwargs={"k": k})
	return retriever.invoke(query)


def decompose_query(user_query: str) -> List[str]:
	prompt = (
    "Analyze the given financial question and break it down into the minimal sub-queries needed to answer it.\n\n"
    "Rules:\n"
    "1. Only extract years that are explicitly mentioned in the question\n"
    "2. Only extract companies that are explicitly mentioned or implied by the question\n"
    "3. If the question asks about 'all companies' or 'which company', include Google, Microsoft, NVIDIA\n"
    "4. If the question is about a specific company and year, don't break it down further\n"
    "5. For comparisons between years, create separate sub-queries for each year mentioned\n"
    "6. Use format: '<Company> <metric> <year>' for each sub-query\n\n"
    "Examples:\n"
    "- 'Which company had the highest operating margin in 2023?' → ['Microsoft operating margin 2023', 'Google operating margin 2023', 'NVIDIA operating margin 2023']\n"
    "- 'How did NVIDIA revenue grow from 2022 to 2023?' → ['NVIDIA revenue 2022', 'NVIDIA revenue 2023']\n"
    "- 'What was Microsoft revenue in 2023?' → [] (don't break down, already specific)\n\n"
    "Return only valid JSON with key 'sub_queries'.\n"
    f"Question: {user_query}\n"
)

	resp = _chat().invoke(prompt)
	text = resp.content if hasattr(resp, "content") else str(resp)
	try:
		obj = json.loads(text)
		subs = obj.get("sub_queries", [])
		if isinstance(subs, list) and subs:
			return [str(s) for s in subs]
		return []
	except Exception:
		return []


def answer_query(vs: FAISS, query: str, k: int = 5) -> Dict[str, Any]:
	sub_queries = decompose_query(query)
	if not sub_queries:
		bundled = [(query, retrieve(vs, query, k=k))]
	else:
		bundled = []
		for subq in sub_queries:
			docs = retrieve(vs, subq, k=k)
			bundled.append((subq, docs))

	context_blocks: List[str] = []
	sources: List[Dict[str, Any]] = []
	for subq, docs in bundled:
		context_lines: List[str] = [f"Sub-question: {subq}"]
		for d in docs:
			chunk_type = d.metadata.get("chunk_type", "")
			source_path = d.metadata.get("source_path", "")
			excerpt = d.page_content[:500]
			context_lines.append(f"[{chunk_type.upper()}: {source_path}]\n{d.page_content}")
			# Extract page number from source_path or use chunk index as page
			page_num = None
			if source_path:
				# Try to extract page from filename or use chunk index
				chunk_index = d.metadata.get("chunk_index", "0")
				if isinstance(chunk_index, str) and chunk_index.isdigit():
					page_num = int(chunk_index) + 1
				elif isinstance(chunk_index, int):
					page_num = chunk_index + 1
			
			sources.append({
				"chunk_type": chunk_type,
				"source_path": source_path,
				"excerpt": excerpt,
				"page": page_num,
			})
		context_blocks.append("\n\n".join(context_lines))

	synthesis_prompt = (
		f"Question: {query}\n\n"
		"Context from 10-K filings:\n" + "\n\n---\n\n".join(context_blocks) + "\n\n"
		"Based on the context above, provide a clear and accurate answer to the question.\n"
		"Then explain your reasoning for how you arrived at this answer.\n\n"
		"Format your response exactly as:\n"
		"Answer: [your answer here]\n"
		"Reasoning: [your reasoning here]"
	)
	resp = _chat().invoke(synthesis_prompt)
	content = resp.content if hasattr(resp, "content") else str(resp)
	
	# Parse the response to extract answer and reasoning
	answer_text = content
	reasoning_text = ""
	
	# Try to extract answer and reasoning from the response
	if "Answer:" in content and "Reasoning:" in content:
		parts = content.split("Reasoning:")
		if len(parts) == 2:
			answer_text = parts[0].replace("Answer:", "").strip()
			reasoning_text = parts[1].strip()
	elif "Answer:" in content:
		answer_text = content.split("Answer:")[1].strip()
	elif "Reasoning:" in content:
		reasoning_text = content.split("Reasoning:")[1].strip()

	return {
		"query": query,
		"answer": answer_text,
		"reasoning": reasoning_text,
		"sub_queries": sub_queries,
		"sources": sources,
	}