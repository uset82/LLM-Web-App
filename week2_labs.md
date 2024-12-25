# Week 2 Laboratory Activities and Project Milestone

## Lab 1: Building an Evaluation Framework
Duration: 2 hours

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
