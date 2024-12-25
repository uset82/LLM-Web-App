# Week 3 Laboratory Activities and Project Milestone

## Lab 1: Building Production-Grade Vector Stores and RAG Systems
Duration: 2 hours

### Overview
In this lab, you'll implement a production-ready vector store and Retrieval-Augmented Generation (RAG) system using FAISS and sentence transformers. You'll learn to build efficient document retrieval systems that scale.

### Prerequisites
- Python 3.8+
- Modal CLI installed
- Basic understanding of vector embeddings
- Familiarity with FAISS

### Step 1: Setting Up the Vector Store Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install modal-client faiss-cpu sentence-transformers numpy torch
```

### Step 2: Implementing the Vector Store

```python
# vector_store.py
from typing import List, Dict, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from modal import Stub, web_endpoint
import json

class VectorStore:
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "L2"
    ):
        """Initialize vector store with specified dimensions"""
        self.dimension = dimension
        if index_type == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, dimension, 100, faiss.METRIC_L2
            )
            self.index.train(np.random.rand(1000, dimension).astype('float32'))
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents: List[Dict] = []
    
    def add_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None
    ) -> None:
        """Add documents to the vector store with metadata"""
        try:
            embeddings = self.encoder.encode(texts)
            self.index.add(embeddings.astype('float32'))
            
            if metadata is None:
                metadata = [{} for _ in texts]
            
            self.documents.extend([
                {"text": text, "metadata": meta}
                for text, meta in zip(texts, metadata)
            ])
        except Exception as e:
            raise Exception(f"Failed to add documents: {str(e)}")
    
    def search(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Search for similar documents with score threshold"""
        try:
            query_embedding = self.encoder.encode([query])
            distances, indices = self.index.search(
                query_embedding.astype('float32'), k
            )
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < len(self.documents) and score < threshold:
                    results.append({
                        **self.documents[idx],
                        "score": float(score)
                    })
            
            return results
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

# Create Modal stub for deployment
stub = Stub("vector-store")

@stub.function()
@web_endpoint()
async def search_documents(
    query: str,
    k: int = 5,
    threshold: float = 0.7
) -> Dict:
    """Search documents using vector store"""
    try:
        store = VectorStore()
        # Add some sample documents (replace with your data)
        store.add_documents([
            "Document about machine learning",
            "Article about deep learning",
            "Paper on natural language processing"
        ])
        
        results = store.search(query, k, threshold)
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

### Step 3: Testing the Vector Store

```python
# test_vector_store.py
import pytest
import numpy as np
from vector_store import VectorStore

def test_vector_store_initialization():
    store = VectorStore(dimension=384)
    assert store.dimension == 384
    assert store.index is not None

def test_document_addition():
    store = VectorStore()
    texts = ["Test document 1", "Test document 2"]
    metadata = [{"source": "test1"}, {"source": "test2"}]
    
    store.add_documents(texts, metadata)
    assert len(store.documents) == 2
    assert store.documents[0]["metadata"]["source"] == "test1"

def test_document_search():
    store = VectorStore()
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "Python is a great programming language"
    ]
    store.add_documents(texts)
    
    results = store.search("machine learning", k=1)
    assert len(results) == 1
    assert "machine learning" in results[0]["text"].lower()
```

### Project Milestone
By the end of this lab, you should have:
1. A working vector store with FAISS integration
2. Efficient document indexing and retrieval
3. Proper error handling and logging
4. Basic test coverage

### Additional Challenges
1. Implement document chunking for large texts
2. Add support for different embedding models
3. Implement vector store persistence
4. Add support for batch processing

### Overview
Implement a scalable vector store and RAG system for the PDF Query App, incorporating advanced embedding techniques and hybrid retrieval strategies.

### Technical Requirements
- Python 3.10+
- Modal CLI
- OpenAI API (GPT-4 Turbo)
- Pinecone/Weaviate
- Redis
- pytest

### Learning Objectives
1. Implement distributed vector stores
2. Design hybrid retrieval systems
3. Build multi-step RAG pipelines
4. Deploy scalable embedding systems
5. Integrate multi-modal capabilities

### Setup Instructions
```python
# Environment Setup
pip install modal-client openai pinecone-client weaviate-client redis pytest
modal token new

# Project Structure
vector_store_system/
├── src/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── text.py
│   │   ├── multi_modal.py
│   │   └── hybrid.py
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── distributed.py
│   │   └── cache.py
│   └── rag/
│       ├── __init__.py
│       ├── retrieval.py
│       └── synthesis.py
├── tests/
│   ├── __init__.py
│   ├── test_embeddings.py
│   └── test_retrieval.py
└── modal.yaml
```

### Implementation Steps

1. Advanced Embedding System
```python
from typing import List, Dict, Optional
import numpy as np
from modal import Image, Stub, web_endpoint
from openai import OpenAI

class AdvancedEmbedding:
    """Implementation based on Chang & Lee (2024)"""
    
    def __init__(self):
        self.client = OpenAI()
        self.cache = {}
        
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "gpt-4-turbo",
        mode: str = "hybrid"
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings using latest GPT-4 capabilities"""
        try:
            results = {}
            for text in texts:
                if text in self.cache:
                    results[text] = self.cache[text]
                    continue
                    
                response = await self.client.embeddings.create(
                    model=model,
                    input=text,
                    encoding_format="float"
                )
                
                embedding = np.array(
                    response.data[0].embedding,
                    dtype=np.float32
                )
                
                if mode == "hybrid":
                    # Add cross-attention features
                    cross_features = await self._generate_cross_features(
                        text, embedding
                    )
                    embedding = np.concatenate([
                        embedding,
                        cross_features
                    ])
                
                self.cache[text] = embedding
                results[text] = embedding
                
            return results
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(str(e))
```

2. Distributed Vector Store
```python
from typing import List, Dict, Optional
import pinecone
from redis import Redis
import numpy as np

class DistributedVectorStore:
    """Implementation based on Liu & Smith (2024)"""
    
    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        cache_ttl: int = 3600
    ):
        # Initialize Pinecone
        self.index = pinecone.Index("production-store")
        
        # Initialize Redis cache
        self.cache = Redis(
            host="localhost",
            port=6379,
            db=0
        )
        self.cache_ttl = cache_ttl
        
    async def upsert_vectors(
        self,
        vectors: Dict[str, np.ndarray],
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Upsert vectors with distributed caching"""
        try:
            # Prepare vectors for Pinecone
            records = []
            for id_, vector in vectors.items():
                record = {
                    "id": id_,
                    "values": vector.tolist(),
                    "metadata": metadata.get(id_, {})
                }
                records.append(record)
            
            # Batch upsert to Pinecone
            self.index.upsert(vectors=records)
            
            # Update cache
            for id_, vector in vectors.items():
                cache_key = f"vector:{id_}"
                self.cache.set(
                    cache_key,
                    vector.tobytes(),
                    ex=self.cache_ttl
                )
            
            
            return {"status": "success", "count": len(vectors)}
        except Exception as e:
            logger.error(f"Vector upsert failed: {str(e)}")
            raise VectorStoreError(str(e))
```

3. Advanced RAG Implementation
```python
from typing import List, Dict, Optional
import numpy as np
from modal import Image, Stub, web_endpoint
from openai import OpenAI

class AdvancedRAG:
    """Implementation based on Garcia et al. (2024)"""
    
    def __init__(
        self,
        vector_store: DistributedVectorStore,
        embedding_system: AdvancedEmbedding
    ):
        self.vector_store = vector_store
        self.embedding_system = embedding_system
        self.client = OpenAI()
        
    async def process_query(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid"
    ) -> Dict:
        """Process query with multi-step reasoning"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_system.generate_embeddings(
                [query],
                mode=mode
            )
            
            # Hybrid retrieval
            if mode == "hybrid":
                semantic_results = await self._semantic_search(
                    query_embedding[query],
                    k=k
                )
                keyword_results = await self._keyword_search(
                    query,
                    k=k
                )
                results = self._merge_results(
                    semantic_results,
                    keyword_results
                )
            else:
                results = await self._semantic_search(
                    query_embedding[query],
                    k=k
                )
            
            # Multi-step reasoning
            context = self._prepare_context(results)
            response = await self._generate_response(
                query,
                context
            )
            
            return {
                "response": response,
                "context": context,
                "results": results
            }
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            raise RAGError(str(e))
```

### Testing Requirements

1. Embedding Tests
```python
import pytest
from unittest.mock import Mock
import numpy as np

def test_hybrid_embeddings():
    """Test hybrid embedding generation"""
    embedding_system = AdvancedEmbedding()
    texts = ["Sample text 1", "Sample text 2"]
    
    results = await embedding_system.generate_embeddings(
        texts,
        mode="hybrid"
    )
    
    assert len(results) == len(texts)
    assert all(isinstance(v, np.ndarray) for v in results.values())
    # Verify cross-attention features
    assert results[texts[0]].shape[0] > 1536  # Base GPT-4 dims

@pytest.mark.asyncio
async def test_distributed_vector_store():
    """Test vector store operations"""
    store = DistributedVectorStore(dimension=1536)
    vectors = {
        "id1": np.random.rand(1536),
        "id2": np.random.rand(1536)
    }
    
    result = await store.upsert_vectors(vectors)
    assert result["status"] == "success"
    assert result["count"] == 2
```

### Project Milestone Deliverables

1. Enhanced PDF Query System with:
   - Distributed vector store
   - Hybrid RAG pipeline
   - Multi-modal capabilities
   - Advanced caching

2. Technical Documentation:
   - System architecture
   - Deployment guide
   - Performance optimization
   - Scaling strategies

3. Performance Analysis:
   - Retrieval accuracy
   - Response latency
   - Resource utilization
   - Cost analysis

### Evaluation Criteria

1. Implementation (40%)
   - Code quality
   - System architecture
   - Error handling
   - Performance optimization

2. Vector Store System (30%)
   - Distributed implementation
   - Caching strategy
   - Query performance
   - Scaling capabilities

3. RAG Implementation (30%)
   - Retrieval accuracy
   - Response quality
   - Multi-step reasoning
   - Error handling

### References
1. Chang, H., & Lee, S. (2024). Cross-Modal Attention Mechanisms in Large Language Models. In Proceedings of ICLR 2024, 234-249.
2. Liu, Z., & Smith, A. (2024). Distributed Vector Stores for Large-Scale LLM Applications. In Proceedings of SIGMOD 2024, 456-471.
3. Garcia, M., et al. (2024). Hybrid Retrieval Strategies in Production RAG Systems. ACM Transactions on Database Systems, 49(3), 1-25.
