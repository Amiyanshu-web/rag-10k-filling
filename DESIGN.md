# RAG System Design Document

## Chunking Strategy

Used Docling for text extraction from PDFs because it handles tables well and preserves all necessary metadata. Docling can extract both text content and structured table data from financial documents, which is crucial for 10-K filings that contain many financial tables.

After extraction, implemented semantic chunking using LangChain's SemanticChunker with a 90% similarity threshold. This approach groups related content together rather than splitting by fixed character counts, which works better for financial documents where related information might span multiple paragraphs.

## Embedding Model Choice

Initially started with Cohere embeddings but hit rate limits on the free plan during development. Switched to sentence-transformers/all-MiniLM-L6-v2 from HuggingFace, which is widely used and reliable. This model provides good semantic understanding for financial text while being free to use and fast to process.

## Query Decomposition Approach

We first send the user query to Cohere with a prompt that decomposes it into multiple sub-queries. The prompt analyzes the question to extract specific companies and years, then creates targeted sub-queries like "Microsoft operating margin 2023" for each relevant combination. This helps retrieve the right data for each specific case before synthesizing the final answer.

## Key Challenges and Decisions

**Tables**: Financial PDFs have many tables. Docling extracts them as markdown so we can search them properly.

**Sources**: We track which document and page each answer comes from so users can verify the information.

**Errors**: Some PDFs might fail to process, so the system keeps working with the ones that succeed.
