# Week 1 Laboratory Activities and Project Milestone

## Lab 1: Production-Grade PDF Query System
Duration: 2 hours

### Overview
In this lab, you'll build a production-ready PDF query system using Modal and OpenAI's GPT models. The system will process PDF documents, extract text, and provide intelligent responses to user queries about the content.

### Prerequisites
- Python 3.8+
- Modal CLI installed
- OpenAI API key configured

### Step 1: Setting Up the Development Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install modal-client openai pypdf
```

### Step 2: Implementing the PDF Processor

```python
# pdf_processor.py
from modal import Image, Stub, web_endpoint
from openai import OpenAI
from pypdf import PdfReader
from io import BytesIO

# Define Modal stub and container image
stub = Stub("pdf-query-system")
image = Image.debian_slim().pip_install(["openai", "pypdf"])

@stub.function(image=image)
def extract_text(pdf_content: bytes) -> str:
    """Extract text from PDF with proper error handling"""
    try:
        pdf = PdfReader(BytesIO(pdf_content))
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

@stub.function()
def query_content(text: str, query: str) -> str:
    """Query PDF content using OpenAI's GPT model"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about PDF documents."},
                {"role": "user", "content": f"Document content: {text}\n\nQuestion: {query}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Query processing failed: {str(e)}")

@stub.function()
@web_endpoint()
async def process_query(pdf_content: bytes, query: str) -> dict:
    """Main endpoint for PDF query system"""
    try:
        # Extract text from PDF
        text = extract_text.remote(pdf_content)
        
        # Process query
        answer = query_content.remote(text, query)
        
        return {
            "status": "success",
            "answer": answer
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```

### Step 3: Testing the System

```python
# test_pdf_processor.py
import pytest
from modal import Stub

def test_extract_text():
    with open("sample.pdf", "rb") as f:
        pdf_content = f.read()
    
    stub = Stub("pdf-query-system")
    text = stub.extract_text.remote(pdf_content)
    assert len(text) > 0

def test_query_content():
    stub = Stub("pdf-query-system")
    response = stub.query_content.remote(
        "The quick brown fox jumps over the lazy dog.",
        "What animal is mentioned?"
    )
    assert "fox" in response.lower()
```

### Project Milestone
By the end of this lab, you should have:
1. A working PDF query system deployed on Modal
2. Proper error handling and logging
3. Basic test coverage
4. Documentation for API endpoints

### Additional Challenges
1. Add support for multiple PDF uploads
2. Implement caching for frequently accessed documents
3. Add rate limiting and usage tracking

### Overview
Build a production-ready PDF query system using Modal and OpenAI's GPT-4, implementing industry best practices for deployment, monitoring, and error handling.

### Technical Requirements
- Python 3.10+
- Modal CLI
- OpenAI API access
- Git for version control
- pytest for testing

### Learning Objectives
1. Implement secure API key management
2. Design robust error handling systems
3. Build efficient streaming responses
4. Deploy serverless applications
5. Monitor application performance

### Setup Instructions
```python
# Environment Setup
pip install modal-client openai pytest python-dotenv
modal token new  # Generate Modal token

# Project Structure
pdf_query/
├── src/
│   ├── __init__.py
│   ├── api.py        # API endpoints
│   ├── processor.py  # PDF processing
│   ├── llm.py       # LLM integration
│   └── utils.py     # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_processor.py
└── modal.yaml       # Modal configuration
```

### Implementation Steps

1. PDF Processing Module
```python
from modal import Image, Stub, method
import fitz  # PyMuPDF

def process_pdf(pdf_path: str) -> str:
    """Extract and preprocess PDF text with proper error handling"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        raise PDFProcessingError(f"Failed to process PDF: {str(e)}")
```

2. LLM Integration
```python
from openai import OpenAI
import asyncio

async def query_document(
    question: str,
    context: str,
    model: str = "gpt-4-1106-preview"
) -> AsyncGenerator[str, None]:
    """Stream responses from GPT-4 with proper error handling"""
    try:
        client = OpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided document context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            stream=True
        )
        async for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"LLM query error: {str(e)}")
        raise LLMQueryError(f"Failed to query LLM: {str(e)}")
```

3. API Implementation
```python
from modal import Image, Stub, web_endpoint
from fastapi import FastAPI, File, UploadFile

stub = Stub("pdf-query-system")
app = FastAPI()

@stub.function()
@web_endpoint(method="POST")
async def query_pdf(
    file: UploadFile = File(...),
    question: str = Form(...)
) -> StreamingResponse:
    """Handle PDF uploads and questions with proper validation"""
    try:
        # Validate file
        if not file.filename.endswith('.pdf'):
            raise ValueError("Only PDF files are supported")
            
        # Process PDF
        text = await process_pdf(file)
        
        # Stream response
        return StreamingResponse(
            query_document(question, text),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Testing Requirements
1. Unit Tests
```python
def test_pdf_processing():
    """Test PDF text extraction"""
    sample_pdf = "tests/data/sample.pdf"
    text = process_pdf(sample_pdf)
    assert len(text) > 0
    assert isinstance(text, str)

@pytest.mark.asyncio
async def test_llm_query():
    """Test LLM response generation"""
    question = "What is the main topic?"
    context = "The document discusses Python programming."
    async for chunk in query_document(question, context):
        assert isinstance(chunk, str)
        assert len(chunk) > 0
```

2. Integration Tests
```python
@pytest.mark.asyncio
async def test_api_endpoint():
    """Test complete API workflow"""
    client = TestClient(app)
    with open("tests/data/sample.pdf", "rb") as f:
        response = client.post(
            "/query_pdf",
            files={"file": f},
            data={"question": "What is this document about?"}
        )
    assert response.status_code == 200
```

### Deployment Instructions
1. Configure Modal
```yaml
# modal.yaml
stub_name: pdf-query-system
image:
  python_version: "3.10"
  apt_install:
    - poppler-utils
  pip_install:
    - PyMuPDF
    - openai
    - python-dotenv
```

2. Deploy Application
```bash
modal deploy pdf_query/src/api.py
```

### Monitoring Setup
1. Implement Logging
```python
import structlog
logger = structlog.get_logger()

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
```

2. Performance Metrics
```python
from prometheus_client import Counter, Histogram

# Define metrics
pdf_processing_time = Histogram(
    'pdf_processing_seconds',
    'Time spent processing PDF'
)
llm_query_time = Histogram(
    'llm_query_seconds',
    'Time spent querying LLM'
)
error_counter = Counter(
    'application_errors_total',
    'Total number of application errors'
)
```

### Project Milestone Deliverables
1. Working PDF Query System
- Implemented error handling
- Streaming responses
- Performance monitoring
- Unit and integration tests

2. Documentation
- API documentation
- Deployment guide
- Monitoring dashboard setup

3. Performance Requirements
- PDF processing < 2s for 10MB files
- Response streaming latency < 100ms
- 99.9% uptime SLA

### Evaluation Criteria
1. Code Quality (40%)
- Error handling
- Type hints
- Documentation
- Test coverage

2. System Design (30%)
- Architecture
- Scalability
- Security

3. Performance (30%)
- Response time
- Resource usage
- Error rates

### References
1. Wang, L., & Chen, H. (2024). Efficient Fine-tuning Strategies for Domain Adaptation in LLMs. In Proceedings of NeurIPS 2024.
2. OpenAI. (2024). Production System Design. OpenAI Documentation.
3. Modal. (2024). Enterprise Deployment Guide. Modal Documentation.
