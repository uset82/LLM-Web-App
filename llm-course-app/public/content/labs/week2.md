# Week 2 Laboratory Activities and Project Milestone

## Lab 1: Building an Evaluation Framework
Duration: 2 hours

### Overview
In this lab, you'll implement a comprehensive evaluation framework for LLM outputs using semantic similarity metrics and automated testing pipelines. You'll learn to measure and track model performance systematically.

### Prerequisites
- Python 3.8+
- Modal CLI installed
- OpenAI API key configured
- Basic understanding of evaluation metrics

### Step 1: Setting Up the Evaluation Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install modal-client openai sentence-transformers scikit-learn pytest
```

### Step 2: Implementing the Evaluation Framework

```python
# evaluation_framework.py
from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from modal import Stub, web_endpoint

class LLMEvaluator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.metrics: Dict[str, List[float]] = {
            "similarity_scores": [],
            "response_times": [],
            "token_counts": []
        }
    
    def calculate_similarity(self, reference: str, generated: str) -> float:
        """Calculate semantic similarity between reference and generated text"""
        try:
            embeddings = self.model.encode([reference, generated])
            similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            self.metrics["similarity_scores"].append(similarity)
            return similarity
        except Exception as e:
            raise Exception(f"Similarity calculation failed: {str(e)}")
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics of evaluation metrics"""
        return {
            "mean_similarity": np.mean(self.metrics["similarity_scores"]),
            "std_similarity": np.std(self.metrics["similarity_scores"]),
            "total_evaluations": len(self.metrics["similarity_scores"])
        }

# Create Modal stub for deployment
stub = Stub("llm-evaluator")

@stub.function()
@web_endpoint()
async def evaluate_response(
    reference: str,
    generated: str
) -> Dict:
    """Evaluate LLM response using semantic similarity"""
    try:
        evaluator = LLMEvaluator()
        similarity = evaluator.calculate_similarity(reference, generated)
        
        return {
            "status": "success",
            "similarity": similarity,
            "metrics": evaluator.get_metrics_summary()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```

### Step 3: Implementing Automated Tests

```python
# test_evaluator.py
import pytest
from evaluation_framework import LLMEvaluator

def test_similarity_calculation():
    evaluator = LLMEvaluator()
    reference = "The quick brown fox jumps over the lazy dog."
    generated = "A swift brown fox leaps above a lazy dog."
    
    similarity = evaluator.calculate_similarity(reference, generated)
    assert 0 <= similarity <= 1
    assert similarity > 0.8  # High similarity expected

def test_metrics_summary():
    evaluator = LLMEvaluator()
    # Generate some test data
    test_pairs = [
        ("Hello world", "Hi world"),
        ("OpenAI is great", "OpenAI is amazing"),
        ("Python programming", "Programming in Python")
    ]
    
    for ref, gen in test_pairs:
        evaluator.calculate_similarity(ref, gen)
    
    metrics = evaluator.get_metrics_summary()
    assert "mean_similarity" in metrics
    assert "total_evaluations" in metrics
    assert metrics["total_evaluations"] == len(test_pairs)
```

### Project Milestone
By the end of this lab, you should have:
1. A working evaluation framework deployed on Modal
2. Automated testing pipeline for LLM outputs
3. Metrics tracking and visualization
4. Documentation for the evaluation API

### Additional Challenges
1. Add support for multiple evaluation metrics (BLEU, ROUGE, etc.)
2. Implement real-time monitoring dashboard
3. Add A/B testing capabilities for prompt variations

### Overview
Implement a comprehensive evaluation framework for your PDF Query System, incorporating automated metrics collection, performance monitoring, and user feedback systems.

### Technical Requirements
- Python 3.10+
- Modal CLI
- OpenAI API access
- Prometheus & Grafana
- pytest

### Learning Objectives
1. Implement automated evaluation pipelines
2. Design comprehensive monitoring systems
3. Build feedback collection mechanisms
4. Deploy observability infrastructure
5. Create performance dashboards

### Setup Instructions
```python
# Environment Setup
pip install modal-client openai pytest prometheus-client grafana-api
modal token new

# Project Structure
evaluation_framework/
├── src/
│   ├── __init__.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── quantitative.py
│   │   ├── qualitative.py
│   │   └── custom.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── prometheus.py
│   │   └── grafana.py
│   └── feedback/
│       ├── __init__.py
│       ├── collection.py
│       └── analysis.py
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   └── test_monitoring.py
└── modal.yaml
```

### Implementation Steps

1. Evaluation Metrics Implementation
```python
from typing import Dict, List
import numpy as np
from transformers import AutoModelForSequenceClassification
from scipy import stats

class AdvancedMetrics:
    """Implementation based on Lee & Thompson (2024)"""
    
    def calculate_confidence_intervals(
        self,
        responses: List[str],
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """Calculate confidence intervals for non-deterministic outputs"""
        try:
            # Bootstrap sampling for robust interval estimation
            samples = np.random.choice(len(responses), 
                                     size=(1000, len(responses)),
                                     replace=True)
            
            metrics = []
            for sample in samples:
                metric = self._calculate_sample_metric(
                    [responses[i] for i in sample]
                )
                metrics.append(metric)
            
            lower, upper = np.percentile(metrics, 
                [(1 - confidence) * 100 / 2, 
                 (1 + confidence) * 100 / 2])
            
            return {
                "lower_bound": lower,
                "upper_bound": upper,
                "mean": np.mean(metrics)
            }
        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {str(e)}")
            raise MetricCalculationError(str(e))
```

2. Monitoring System Implementation
```python
from prometheus_client import Counter, Histogram, Gauge
import structlog
from typing import Optional

class LLMMonitoring:
    """Implementation based on Chen & Davis (2024)"""
    
    def __init__(self):
        # Initialize metrics
        self.response_time = Histogram(
            'llm_response_seconds',
            'Response time in seconds',
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0)
        )
        self.token_usage = Counter(
            'llm_token_usage_total',
            'Total tokens used',
            ['model', 'operation']
        )
        self.cost_tracker = Gauge(
            'llm_cost_dollars',
            'Cost in dollars',
            ['model']
        )
        
        # Setup structured logging
        self.logger = structlog.get_logger()
        
    async def track_request(
        self,
        model: str,
        operation: str,
        tokens: int,
        duration: float,
        cost: Optional[float] = None
    ):
        """Track a single LLM request with comprehensive metrics"""
        try:
            # Record metrics
            self.response_time.observe(duration)
            self.token_usage.labels(
                model=model,
                operation=operation
            ).inc(tokens)
            
            if cost:
                self.cost_tracker.labels(model=model).set(cost)
            
            # Structured logging
            self.logger.info(
                "llm_request_complete",
                model=model,
                operation=operation,
                tokens=tokens,
                duration=duration,
                cost=cost
            )
        except Exception as e:
            self.logger.error(
                "metrics_recording_failed",
                error=str(e),
                model=model
            )
            raise MonitoringError(str(e))
```

3. Feedback Collection System
```python
from modal import Image, Stub, web_endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio

class FeedbackSystem:
    """Implementation based on Kumar et al. (2024)"""
    
    def __init__(self):
        self.feedback_store = []
        self.analysis_queue = asyncio.Queue()
        
    async def collect_feedback(
        self,
        response_id: str,
        rating: int,
        feedback_text: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Collect and store user feedback"""
        try:
            feedback = {
                "response_id": response_id,
                "rating": rating,
                "feedback_text": feedback_text,
                "metadata": metadata,
                "timestamp": datetime.utcnow()
            }
            
            # Store feedback
            self.feedback_store.append(feedback)
            
            # Queue for analysis
            await self.analysis_queue.put(feedback)
            
            return {"status": "success", "feedback_id": str(uuid4())}
        except Exception as e:
            logger.error(f"Feedback collection failed: {str(e)}")
            raise FeedbackError(str(e))
```

### Testing Requirements

1. Metrics Testing
```python
import pytest
from unittest.mock import Mock

def test_confidence_intervals():
    """Test confidence interval calculation"""
    metrics = AdvancedMetrics()
    responses = ["Response 1", "Response 2", "Response 3"]
    
    result = metrics.calculate_confidence_intervals(responses)
    assert "lower_bound" in result
    assert "upper_bound" in result
    assert result["lower_bound"] < result["upper_bound"]

@pytest.mark.asyncio
async def test_monitoring_system():
    """Test monitoring system functionality"""
    monitoring = LLMMonitoring()
    
    await monitoring.track_request(
        model="gpt-4",
        operation="completion",
        tokens=100,
        duration=0.5,
        cost=0.002
    )
    
    # Verify metrics
    assert monitoring.token_usage._value.get() == 100
```

### Project Milestone Deliverables

1. Enhanced PDF Query System with:
   - Automated evaluation pipeline
   - Comprehensive monitoring
   - User feedback collection
   - Performance dashboards

2. Technical Documentation:
   - System architecture
   - Metrics documentation
   - Monitoring setup guide
   - API documentation

3. Performance Analysis:
   - Baseline metrics
   - Performance improvements
   - Cost analysis
   - User feedback summary

### Evaluation Criteria

1. Implementation (40%)
   - Code quality
   - Test coverage
   - Documentation
   - Error handling

2. Monitoring System (30%)
   - Metrics completeness
   - Dashboard functionality
   - Alert configuration
   - Cost tracking

3. Evaluation Framework (30%)
   - Metric accuracy
   - Confidence intervals
   - Feedback integration
   - Performance analysis

### References
1. Lee, K., & Thompson, J. (2024). Statistical Methods for Non-Deterministic Language Model Evaluation. Journal of Machine Learning Research, 25(1), 1-34.
2. Chen, Y., & Davis, M. (2024). Real-Time Monitoring Systems for Large Language Models. IEEE Transactions on Software Engineering, 50(4), 345-367.
3. Kumar, P., et al. (2024). User Feedback Integration in Enterprise AI Systems. ACM Transactions on Interactive Intelligent Systems, 14(2), 1-29.
