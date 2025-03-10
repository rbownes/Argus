# Advanced LLM Evaluation Features

This extension adds two powerful capabilities to the LLM evaluation framework:

1. **LLM-as-Judge Evaluation**: Use powerful LLMs to evaluate the quality, factuality, and coherence of other LLMs' responses with detailed, multi-criteria assessments.

2. **Standardized Benchmarks**: Evaluate LLMs against industry-standard benchmarks like MMLU, TruthfulQA, and GSM8K with consistent metrics and flexible configurations.

## LLM-as-Judge Evaluator

The LLM-as-Judge evaluator uses a stronger LLM (like GPT-4 or Claude) to evaluate the responses of other LLMs based on specific criteria.

### Features

- **Multiple Evaluation Types**: Supports various evaluation dimensions including quality, factuality, coherence, relevance, consistency, and toxicity.
- **Customizable Criteria**: Each evaluation type has default criteria, but you can provide custom criteria for specific use cases.
- **Detailed Feedback**: Returns structured evaluations with overall scores, criteria-specific scores, explanations, strengths, weaknesses, and improvement suggestions.
- **Consistent Scoring**: Normalizes scores from 0-10 to 0-1 scale for consistency with other evaluators.

### Usage Example

```python
from llm_eval.services.evaluation_service import LLMJudgeEvaluator, EvaluationType
from llm_eval.services.llm_service import LiteLLMService

# Initialize the LLM service
llm_service = LiteLLMService()

# Create the LLM-as-Judge evaluator
judge_evaluator = LLMJudgeEvaluator(
    llm_service=llm_service,
    judge_model="gpt-4-turbo",
    evaluation_type=EvaluationType.QUALITY,
    temperature=0.1
)

# Evaluate a response
result = await judge_evaluator.evaluate(response)
if result.is_ok:
    evaluation = result.unwrap()
    print(f"Score: {evaluation.score}")
    print(f"Explanation: {evaluation.explanation}")
    
    # Access detailed criteria scores
    criteria_scores = evaluation.metadata["criteria_scores"]
    for criterion, data in criteria_scores.items():
        print(f"{criterion}: {data['score']} - {data['reasoning']}")
```

## Standardized Benchmarks

The benchmark service enables evaluating LLMs against popular benchmarks with standardized metrics.

### Included Benchmarks

1. **MMLU (Massive Multitask Language Understanding)**
   - Tests knowledge across 57 subjects including STEM, humanities, and social sciences
   - Provides subject-specific metrics with optional few-shot examples

2. **TruthfulQA**
   - Tests LLM truthfulness on questions where models might generate incorrect answers
   - Supports both generation and multiple-choice evaluation modes

3. **GSM8K (Grade School Math)**
   - Tests multi-step mathematical reasoning abilities
   - Evaluates both correctness and reasoning process

### Usage Example

```python
from llm_eval.services.benchmark_service import BenchmarkService
from llm_eval.services.llm_service import LiteLLMService

# Initialize services
llm_service = LiteLLMService()
benchmark_service = BenchmarkService(llm_service=llm_service)

# Run a benchmark on a single model
result = await benchmark_service.run_benchmark(
    benchmark_id="mmlu",
    model_name="gpt-3.5-turbo",
    config={
        "sample_limit": 20,
        "categories": ["high_school_mathematics", "philosophy"],
        "shuffle": True
    }
)

# Compare multiple models
comparison = await benchmark_service.run_benchmark_comparison(
    benchmark_id="truthfulqa",
    model_names=["gpt-3.5-turbo", "claude-3-opus"],
    config={"eval_mode": "generation"}
)
```

## Integration Example

The `examples/judge_benchmarks/llm_judge_benchmark_example.py` script demonstrates how to combine the LLM-as-Judge evaluator with standardized benchmarks for comprehensive evaluation:

1. Run benchmark evaluations on multiple models
2. Use the LLM-as-Judge evaluator to assess the quality of responses
3. Compare objective benchmark metrics with subjective judge evaluations
4. Save detailed results for further analysis

## Setting Up Benchmark Data

Each benchmark requires specific data files:

- **MMLU**: Create a directory `data/mmlu/` with CSV files for each subject
- **TruthfulQA**: Create a JSON file at `data/truthfulqa/truthfulqa.json`
- **GSM8K**: Create a JSONL file at `data/gsm8k/gsm8k.jsonl`

Refer to each benchmark's documentation for specific data format requirements.

## Extending the System

### Adding New Benchmarks

1. Create a new class inheriting from `BaseBenchmark`
2. Implement the required methods (load_samples, create_prompt, evaluate_response, calculate_benchmark_metrics)
3. Register your benchmark with the BenchmarkService

### Custom Evaluation Criteria

You can customize the LLM-as-Judge evaluator with your own evaluation criteria:

```python
custom_criteria = [
    "Accuracy of technical information",
    "Use of domain-specific terminology",
    "Readability for the target audience",
    "Coverage of key concepts"
]

judge_evaluator = LLMJudgeEvaluator(
    llm_service=llm_service,
    judge_model="claude-3-opus",
    evaluation_type=EvaluationType.QUALITY,
    criteria=custom_criteria
)
```
