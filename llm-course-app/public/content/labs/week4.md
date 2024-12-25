# Week 4 Laboratory Activities and Project Milestone

## Lab 1: Advanced Prompt Engineering and Fine-Tuning
Duration: 2 hours

### Overview
In this lab, you'll implement efficient fine-tuning strategies using LoRA (Low-Rank Adaptation) and deploy the fine-tuned models using Modal. You'll learn to optimize model performance and manage computational resources effectively.

### Prerequisites
- Python 3.8+
- Modal CLI installed
- Basic understanding of model fine-tuning
- Familiarity with PyTorch and Transformers

### Step 1: Setting Up the Fine-tuning Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install modal-client torch transformers peft datasets accelerate
```

### Step 2: Implementing LoRA Fine-tuning

```python
# fine_tuning.py
from typing import Dict, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from modal import Stub, Image, web_endpoint

class ModelTrainer:
    def __init__(
        self,
        base_model: str,
        rank: int = 8,
        alpha: float = 32.0
    ):
        """Initialize model trainer with LoRA configuration"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)
    
    def prepare_dataset(
        self,
        texts: List[str],
        labels: List[str]
    ) -> Dataset:
        """Prepare dataset for fine-tuning"""
        return Dataset.from_dict({
            "text": texts,
            "label": labels
        })
    
    def train(
        self,
        dataset: Dataset,
        output_dir: str = "fine_tuned_model",
        num_epochs: int = 3
    ):
        """Fine-tune the model using LoRA"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            learning_rate=3e-4
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        trainer.train()
        
    def generate(
        self,
        prompt: str,
        max_length: int = 100
    ) -> str:
        """Generate text using fine-tuned model"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create Modal stub and container image
stub = Stub("fine-tuning")
image = Image.debian_slim().pip_install([
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets"
])

@stub.function(image=image, gpu="A10G")
@web_endpoint()
async def fine_tune_model(
    texts: List[str],
    labels: List[str]
) -> Dict:
    """Fine-tune model with provided data"""
    try:
        trainer = ModelTrainer("gpt2")
        dataset = trainer.prepare_dataset(texts, labels)
        trainer.train(dataset)
        
        return {
            "status": "success",
            "message": "Model fine-tuned successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```

### Step 3: Testing the Fine-tuning Pipeline

```python
# test_fine_tuning.py
import pytest
from fine_tuning import ModelTrainer
from datasets import Dataset

def test_model_initialization():
    trainer = ModelTrainer("gpt2")
    assert trainer.model is not None
    assert trainer.tokenizer is not None

def test_dataset_preparation():
    trainer = ModelTrainer("gpt2")
    texts = ["Sample text 1", "Sample text 2"]
    labels = ["Label 1", "Label 2"]
    
    dataset = trainer.prepare_dataset(texts, labels)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2

def test_text_generation():
    trainer = ModelTrainer("gpt2")
    prompt = "Once upon a time"
    
    
    generated_text = trainer.generate(prompt, max_length=50)
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0
```

### Project Milestone
By the end of this lab, you should have:
1. A working fine-tuning pipeline using LoRA
2. Efficient model deployment on Modal
3. Basic evaluation metrics
4. Documentation for the fine-tuning API

### Additional Challenges
1. Implement different fine-tuning strategies (P-tuning, Prefix-tuning)
2. Add support for different model architectures
3. Implement model compression techniques
4. Add support for custom datasets

### Overview
Implement advanced prompt engineering techniques and fine-tuning strategies for the PDF Query App, incorporating the latest GPT-4 Turbo capabilities and Modal deployment features.

### Technical Requirements
- Python 3.10+
- Modal CLI
- OpenAI API (GPT-4 Turbo)
- PyTorch
- Transformers
- pytest

### Learning Objectives
1. Implement advanced prompt engineering
2. Design fine-tuning pipelines
3. Deploy production systems
4. Build multi-agent architectures
5. Optimize resource usage

### Setup Instructions
```python
# Environment Setup
pip install modal-client openai torch transformers pytest wandb
modal token new

# Project Structure
production_system/
├── src/
│   ├── __init__.py
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── engineering.py
│   │   ├── optimization.py
│   │   └── evaluation.py
│   ├── fine_tuning/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── training.py
│   └── deployment/
│       ├── __init__.py
│       ├── infrastructure.py
│       └── monitoring.py
├── tests/
│   ├── __init__.py
│   ├── test_prompts.py
│   └── test_deployment.py
└── modal.yaml
```

### Implementation Steps

1. Advanced Prompt Engineering
```python
from typing import Dict, List, Optional
from modal import Image, Stub, web_endpoint
from openai import OpenAI
import json

class AdvancedPromptEngine:
    """Implementation based on Kim & Park (2024)"""
    
    def __init__(self):
        self.client = OpenAI()
        self.cache = {}
        
    async def optimize_prompt(
        self,
        base_prompt: str,
        examples: List[Dict],
        mode: str = "chain-of-thought"
    ) -> Dict:
        """Optimize prompts using latest GPT-4 capabilities"""
        try:
            # Prepare system message
            system_message = self._build_system_message(
                mode=mode,
                examples=examples
            )
            
            # Generate optimized prompt
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": base_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json"}
            )
            
            optimized = json.loads(
                response.choices[0].message.content
            )
            
            # Validate and test
            metrics = await self._evaluate_prompt(
                optimized["prompt"],
                examples
            )
            
            return {
                "optimized_prompt": optimized["prompt"],
                "improvements": optimized["improvements"],
                "metrics": metrics
            }
        except Exception as e:
            logger.error(f"Prompt optimization failed: {str(e)}")
            raise PromptError(str(e))
```

2. Fine-Tuning Pipeline
```python
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import wandb

class FineTuningPipeline:
    """Implementation based on Chen et al. (2024)"""
    
    def __init__(
        self,
        model_name: str,
        training_args: Optional[Dict] = None
    ):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        
        # Setup training arguments
        self.training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            **training_args or {}
        )
        
    async def prepare_data(
        self,
        raw_data: List[Dict],
        validation_split: float = 0.2
    ) -> Dict:
        """Prepare data for fine-tuning"""
        try:
            # Process and tokenize data
            processed = []
            for item in raw_data:
                tokens = self.tokenizer(
                    item["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                processed.append({
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "labels": item["labels"]
                })
            
            
            # Split data
            split_idx = int(len(processed) * (1 - validation_split))
            train_data = processed[:split_idx]
            val_data = processed[split_idx:]
            
            return {
                "train": train_data,
                "validation": val_data
            }
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise DataError(str(e))
```

3. Production Deployment
```python
from typing import Dict, Optional
from modal import Image, Stub, web_endpoint
import prometheus_client as prom
import structlog

class ProductionDeployment:
    """Implementation based on Wilson & Brown (2024)"""
    
    def __init__(
        self,
        config: Optional[Dict] = None
    ):
        self.logger = structlog.get_logger()
        
        # Initialize metrics
        self.request_latency = prom.Histogram(
            'request_latency_seconds',
            'Request latency in seconds'
        )
        self.error_counter = prom.Counter(
            'error_total',
            'Total number of errors'
        )
        
    async def deploy_service(
        self,
        model_path: str,
        scaling_config: Dict
    ) -> Dict:
        """Deploy service with auto-scaling"""
        try:
            # Create Modal stub
            stub = Stub("production-service")
            
            # Configure image
            image = Image.debian_slim().pip_install(
                "torch",
                "transformers",
                "prometheus_client"
            )
            
            # Define web endpoint
            @stub.function(
                image=image,
                gpu="A100",
                timeout=600
            )
            @web_endpoint(method="POST")
            async def predict(self, request: Dict) -> Dict:
                with self.request_latency.time():
                    try:
                        result = await self._process_request(
                            request
                        )
                        return result
                    except Exception as e:
                        self.error_counter.inc()
                        raise
            
            # Deploy
            stub.deploy()
            
            return {
                "status": "deployed",
                "endpoint": predict.url
            }
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            raise DeploymentError(str(e))
```

### Testing Requirements

1. Prompt Testing
```python
import pytest
from unittest.mock import Mock

def test_prompt_optimization():
    """Test prompt optimization"""
    engine = AdvancedPromptEngine()
    base_prompt = "Summarize this document"
    examples = [
        {"input": "doc1", "output": "summary1"},
        {"input": "doc2", "output": "summary2"}
    ]
    
    result = await engine.optimize_prompt(
        base_prompt,
        examples,
        mode="chain-of-thought"
    )
    
    assert "optimized_prompt" in result
    assert "improvements" in result
    assert "metrics" in result
    assert result["metrics"]["performance"] > 0.8

@pytest.mark.asyncio
async def test_deployment():
    """Test production deployment"""
    deployment = ProductionDeployment()
    
    result = await deployment.deploy_service(
        model_path="./model",
        scaling_config={"min_replicas": 2}
    )
    
    assert result["status"] == "deployed"
    assert "endpoint" in result
```

### Project Milestone Deliverables

1. Production-Ready PDF Query System with:
   - Optimized prompts
   - Fine-tuned models
   - Auto-scaling deployment
   - Multi-agent capabilities

2. Technical Documentation:
   - System architecture
   - Deployment guide
   - Performance optimization
   - Scaling strategies

3. Performance Analysis:
   - Response quality
   - System latency
   - Resource utilization
   - Cost analysis

### Evaluation Criteria

1. Implementation (40%)
   - Code quality
   - System architecture
   - Error handling
   - Performance optimization

2. Deployment (30%)
   - Infrastructure setup
   - Scaling strategy
   - Monitoring system
   - Resource management

3. Documentation (30%)
   - Architecture diagrams
   - API documentation
   - Deployment guide
   - Performance analysis

### References
1. Kim, S., & Park, J. (2024). Advanced Prompt Engineering in Production Systems. In Proceedings of ACL 2024, 234-249.
2. Chen, Y., et al. (2024). Parameter-Efficient Fine-Tuning for Large Language Models. Nature Machine Intelligence, 6(4), 345-360.
3. Wilson, R., & Brown, A. (2024). Scalable Deployment Architectures for LLM Applications. IEEE Transactions on Software Engineering, 50(6), 789-804.
