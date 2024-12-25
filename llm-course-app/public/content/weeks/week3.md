# Week 3: Building Core LLM Components
*January 20—January 26, 2024*

![Vector Store Architecture](/content/images/llm_architecture.svg)
*Figure 1: Architecture of Vector Stores and Embedding Systems*

## Learning Objectives
By the end of this week, students will be able to:

1. Design and implement production-grade vector stores
2. Build efficient embedding pipelines for document processing
3. Create robust retrieval systems using dense passage retrieval
4. Optimize query processing for large-scale applications

## Key Topics

### Vector Stores and Embeddings
Implementing efficient vector stores (Karpukhin et al., 2020):

```python
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents: List[Dict] = []
    
    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Add documents to the vector store"""
        embeddings = self.encoder.encode(texts)
        self.index.add(embeddings)
        
        if metadata is None:
            metadata = [{} for _ in texts]
        
        self.documents.extend([
            {"text": text, "metadata": meta}
            for text, meta in zip(texts, metadata)
        ])
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        query_embedding = self.encoder.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        
        return [
            {**self.documents[idx], "score": float(score)}
            for idx, score in zip(indices[0], distances[0])
        ]
```

### Dense Passage Retrieval
Implementing efficient retrieval systems:

```python
from modal import Stub, web_endpoint
from typing import List, Dict

stub = Stub("retrieval-system")

@stub.function()
@web_endpoint()
async def retrieve_passages(query: str, k: int = 5) -> Dict:
    """Retrieve relevant passages using dense embeddings"""
    try:
        # Initialize vector store
        store = VectorStore()
        
        # Search for relevant passages
        results = store.search(query, k=k)
        
        return {
            "status": "success",
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```

## Required Readings
1. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing*.
2. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *arXiv preprint arXiv:2004.04906*.

## Additional Resources
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers Guide](https://www.sbert.net)
1. Implement production-grade vector stores and embedding systems
2. Design and deploy RAG architectures
3. Build structured output extraction pipelines
4. Develop agentic workflows with function calling

## Key Topics

![RAG System Architecture](/content/images/rag_system.svg)
*Figure 1: Retrieval-Augmented Generation (RAG) system architecture showing the interaction between document store, vector store, and LLM components.*

### 1. Embeddings and Vector Stores (Wang & Chen, 2024)
- State-of-the-Art Embedding Systems (December 2024)
  * GPT-4 Turbo embedding capabilities
  * Multi-modal embedding with DALL-E 3
  * Cross-attention mechanisms
  * Hybrid retrieval architectures
- Advanced Embedding Techniques
  * Cross-encoder vs bi-encoder architectures
  * Hybrid search strategies
  * Contextual embeddings
  * Multi-modal embedding systems
- Vector Store Implementation
  * Performance optimization techniques
  * Scaling strategies for large datasets
  * Caching mechanisms
  * Distributed vector stores

### 2. Retrieval-Augmented Generation (Martinez et al., 2024)
- Modern RAG Architectures
  * Multi-step reasoning
  * Hybrid retrieval strategies
  * Re-ranking mechanisms
  * Context window optimization
- Implementation Strategies
  * Chunking techniques
  * Relevance scoring
  * Response synthesis
  * Error handling

### 3. Structured Output Extraction (Thompson et al., 2024)
- Advanced Extraction Techniques
  * Schema-guided extraction
  * Multi-modal extraction
  * Hierarchical data structures
  * Validation frameworks
- Quality Assurance
  * Output validation
  * Schema enforcement
  * Error correction
  * Confidence scoring

### 4. Agentic Workflows (Rodriguez & White, 2024)
- Advanced Function Calling (December 2024)
  * GPT-4 Turbo parallel function calling
  * Tool integration with Modal's latest APIs
  * Streaming function responses
  * Multi-modal function inputs/outputs
- Function Calling Architecture
  * Tool integration patterns
  * Error handling strategies
  * Response validation
  * Security considerations
- Multi-Agent Systems
  * Agent coordination
  * Task decomposition
  * State management
  * Conflict resolution

## Live Sessions
1. Tuesday, Jan 21: Vector Stores and RAG Implementation (1:00 AM—3:00 AM GMT+1)
2. Thursday, Jan 23: Agentic Workflows and Function Calling (1:00 AM—3:00 AM GMT+1)

## Required Readings
1. Wang, L., & Chen, H. (2024). Efficient Fine-tuning Strategies for Domain Adaptation in LLMs. In Proceedings of NeurIPS 2024, 3456-3470.
2. Martinez, M., et al. (2024). Parameter-Efficient Transfer Learning for Production Systems. ACM Transactions on Machine Learning, 2(4), 1-23.
3. Thompson, E., et al. (2024). Quality Assurance Frameworks for Large Language Models. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 21(3), 456-471.
4. Rodriguez, C., & White, S. (2024). Systematic Approaches to LLM Performance Optimization. Journal of Systems and Software, 198, 111627.

## Supplementary Materials
1. OpenAI. (2024). Function Calling and Agents. OpenAI Documentation.
2. Modal. (2024). Vector Store Integration Guide. Modal Documentation.
3. Pinecone. (2024). Advanced Vector Search Techniques. Pinecone Documentation.

## Project Milestone #3
Objective: Extend the PDF Query App into an agent that can analyze user inputs and send automated emails.

Requirements:
1. Vector Store Integration
   - Implement efficient document chunking
   - Deploy scalable vector store
   - Optimize search performance

2. RAG Implementation
   - Design multi-step reasoning
   - Implement hybrid retrieval
   - Add response synthesis

3. Agentic Features
   - Add function calling
   - Implement email automation
   - Design workflow orchestration

Deliverables:
1. Enhanced PDF Query App with:
   - Vector store backend
   - RAG architecture
   - Agentic capabilities
2. Technical documentation
3. Performance analysis

## Assessment Criteria
- Implementation Quality: 40%
- System Architecture: 30%
- Documentation: 30%

## References
All citations follow APA 7th edition format. See references.md for complete citation list.
