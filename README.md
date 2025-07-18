# Tree of Counterfactuals Framework for Automated Fact-Checking (ToCF)

An innovative fact-checking system that leverages counterfactual reasoning and Large Language Models to enhance the accuracy and interpretability of automated fact verification.

## Overview

The ToCF framework introduces a novel approach to automated fact-checking by generating tree-structured counterfactuals that provide multiple perspectives on claims. This methodology improves fact-checking accuracy through weighted aggregation of predictions from original statements and their counterfactual variants.

## Key Features

- **Counterfactual Generation**: Tree-structured generation of alternative statements using LLMs
- **Enhanced Fact-Checking**: Weighted aggregation combining original and counterfactual predictions
- **Vietnamese Language Support**: Optimized prompts and processing for Vietnamese text
- **Memory Optimization**: 4-bit quantization and gradient checkpointing for efficient processing
- **Comprehensive Evaluation**: Full metrics suite with visualization capabilities
- **Production-Ready**: Robust error handling, caching, and batch processing

## System Architecture

The ToCF system consists of two main components:

### 1. Counterfactual Generation (`generate_ToCF.py`)
- Generates counterfactual statements using LLMs
- Supports both individual and tree-structured generation
- Implements batch processing for efficiency
- Provides intelligent caching and retry mechanisms

### 2. Fact-Checking (`factchecking_ToCF.py`)
- Performs stance detection using counterfactuals and evidence
- Implements weighted aggregation with confidence thresholding
- Supports multiple evidence formats (JSON, lists, strings)
- Generates comprehensive evaluation reports

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- HuggingFace account with access token

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

### Model Access

Ensure you have access to the required models:
- Default counterfactual generation: `Qwen/Qwen2.5-72B-Instruct`
- Default fact-checking: `ura-hcmut/ura-llama-13b`

## Usage

### 1. Generate Counterfactuals

```bash
python generate_ToCF.py input_file.csv output_file.csv --token YOUR_HF_TOKEN [OPTIONS]
```

**Basic Usage:**
```bash
python generate_ToCF.py statements.csv counterfactuals.csv --token hf_xxxxx
```

**Advanced Options:**
```bash
python generate_ToCF.py statements.csv counterfactuals.csv \
    --token hf_xxxxx \
    --model Qwen/Qwen2.5-72B-Instruct \
    --statement-column "Statement" \
    --use-tree \
    --tree-depth 2 \
    --batch-size 4 \
    --test-mode
```

### 2. Perform Fact-Checking

```bash
python factchecking_ToCF.py counterfactuals.csv fact_check_data.csv results.csv --token YOUR_HF_TOKEN [OPTIONS]
```

**Basic Usage:**
```bash
python factchecking_ToCF.py counterfactuals.csv dataset.csv results.csv --token hf_xxxxx
```

**Advanced Options:**
```bash
python factchecking_ToCF.py counterfactuals.csv dataset.csv results.csv \
    --token hf_xxxxx \
    --model ura-hcmut/ura-llama-13b \
    --statement-column "Statement" \
    --context-column "Context" \
    --evidence-column "Evidence_List" \
    --label-column "labels" \
    --aggregation-threshold 0.6 \
    --test-mode
```

## Command Line Arguments

### generate_ToCF.py

| Argument | Description | Default |
|----------|-------------|---------|
| `input_file` | Input CSV file with statements | Required |
| `output_file` | Output CSV file for counterfactuals | Required |
| `--token` | HuggingFace access token | Required |
| `--model` | Model name for generation | `Qwen/Qwen2.5-72B-Instruct` |
| `--statement-column` | Column name for statements | `Statement` |
| `--use-tree` | Enable tree generation mode | False |
| `--tree-depth` | Depth for tree generation | 2 |
| `--batch-size` | Batch size for processing | 4 |
| `--test-mode` | Test with first 5 statements | False |
| `--disable-cache` | Disable result caching | False |

### factchecking_ToCF.py

| Argument | Description | Default |
|----------|-------------|---------|
| `counterfactuals_file` | CSV file with counterfactuals | Required |
| `fact_check_file` | CSV file with fact-check data | Required |
| `output_file` | Output CSV file for results | Required |
| `--token` | HuggingFace access token | Required |
| `--model` | Model name for fact-checking | `ura-hcmut/ura-llama-13b` |
| `--statement-column` | Statement column name | `Statement` |
| `--context-column` | Context column name | `Context` |
| `--evidence-column` | Evidence column name | `Evidence_List` |
| `--label-column` | True label column name | `labels` |
| `--aggregation-threshold` | Threshold for label aggregation | 0.6 |
| `--batch-size` | Batch size for processing | 1 |
| `--test-mode` | Test with first 5 statements | False |
| `--evaluate-only` | Only evaluate existing results | False |
| `--disable-cache` | Disable result caching | False |

## Data Format

### Input Data Format

**Statements File (for counterfactual generation):**
```csv
Statement
"Company increased profits by 20% in 2023"
"The new policy was implemented last month"
```

**Fact-Check Dataset:**
```csv
Statement,Context,Evidence_List,labels
"Company increased profits by 20% in 2023","Financial report context","[""Evidence 1"", ""Evidence 2""]",0
"The new policy was implemented last month","Policy context","[""Policy document""]",1
```

### Output Data Format

**Counterfactuals Output:**
```csv
Statement,counterfactual_1,counterfactual_2,counterfactual_count,status
"Original statement","Counterfactual 1","Counterfactual 2",2,success
```

**Fact-Check Results:**
```csv
Statement,Context,Evidence_List,labels,predicted_label,processing_status
"Original statement","Context","Evidence",0,1,success
```

## Label Mapping

The system uses the following label mapping:
- **0 (Support)**: Evidence supports the statement
- **1 (Refuted)**: Evidence refutes the statement  
- **2 (NEI)**: Not Enough Information to determine

## Performance Optimization

### Memory Management
- **4-bit Quantization**: Reduces model memory usage by ~75%
- **Gradient Checkpointing**: Trades computation for memory efficiency
- **Periodic GPU Cache Clearing**: Prevents memory accumulation

### Processing Efficiency
- **Intelligent Caching**: MD5-based cache keys for result reuse
- **Batch Processing**: Processes multiple statements simultaneously
- **Cache-Aware Batching**: Only processes uncached statements

### Error Handling
- **Retry Logic**: Exponential backoff for failed generations
- **Graceful Degradation**: Continues processing on individual failures
- **Comprehensive Logging**: Multi-level logging with progress tracking

## Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and micro F1 scores
- **Precision/Recall**: Per-class and averaged metrics
- **Confusion Matrix**: Visual representation of classification results
- **Processing Statistics**: Success rates, cache utilization, error tracking

## Example Workflow

1. **Prepare your dataset** with statements to fact-check
2. **Generate counterfactuals** using `generate_ToCF.py`
3. **Perform fact-checking** using `factchecking_ToCF.py` with evidence
4. **Analyze results** using the generated metrics and visualizations

```bash
# Step 1: Generate counterfactuals
python generate_ToCF.py statements.csv counterfactuals.csv --token hf_xxxxx

# Step 2: Perform fact-checking
python factchecking_ToCF.py counterfactuals.csv dataset.csv results.csv --token hf_xxxxx

# Step 3: Evaluate results (if ground truth labels available)
python factchecking_ToCF.py counterfactuals.csv dataset.csv results.csv --token hf_xxxxx --evaluate-only
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or enable additional optimizations
2. **Model Access Denied**: Ensure your HuggingFace token has proper permissions
3. **Slow Processing**: Enable caching and use appropriate batch sizes
4. **Poor Counterfactual Quality**: Adjust generation parameters or try different models

### Performance Tips

- Use `--test-mode` for initial testing with small datasets
- Enable caching for repeated experiments
- Monitor GPU memory usage and adjust batch sizes accordingly
- Use tree generation for more comprehensive counterfactual coverage
