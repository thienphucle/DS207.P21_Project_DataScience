# Enhanced Tree of Counterfactual (ToCF) Fact-Checking System

This repository contains an improved implementation of the Tree of Counterfactual (ToCF) approach for automated fact-checking. The system generates counterfactual statements and uses them to enhance fact-checking accuracy through multi-perspective reasoning.

## 🚀 Key Improvements Made

### 1. **Enhanced Prompting Strategy**
- **English Prompts**: Switched from Vietnamese to English prompts for better model compatibility
- **Structured Format**: Clear, step-by-step instructions with explicit formatting requirements
- **Better Context**: Added reasoning process guidance and examples for counterfactual generation

### 2. **Robust Label Extraction**
- **Multiple Patterns**: Supports various output formats ([LABEL: X], [Label: X], etc.)
- **Fallback Logic**: Multiple extraction strategies if primary pattern fails
- **Case Insensitive**: Handles different capitalization patterns

### 3. **Advanced Aggregation Strategy**
- **Weighted Voting**: Original statement gets double weight in decision making
- **Confidence Levels**: Different thresholds for high/medium/low confidence decisions
- **Consistency Checking**: Analyzes agreement between counterfactuals
- **Smart Fallback**: Defaults to NEI when high disagreement detected

### 4. **Improved Counterfactual Generation**
- **Better Validation**: Checks for actual contradiction with original statement
- **Quality Filtering**: Ensures counterfactuals are realistic and well-formed
- **Multiple Extraction Patterns**: Robust parsing of generated counterfactuals
- **Contradiction Detection**: Heuristic to verify counterfactuals actually contradict original

### 5. **Optimized Generation Parameters**
- **Fact-Checking**: Lower temperature (0.1) for consistent classification
- **Counterfactual Generation**: Balanced parameters for creative but controlled generation
- **Repetition Penalty**: Prevents repetitive outputs
- **Early Stopping**: Improves efficiency

## 📁 Files Overview

### `generate_ToCF.py`
Generates counterfactual statements for input claims.

**Key Features:**
- Batch processing for efficiency
- Tree-based generation (hierarchical counterfactuals)
- Improved prompt engineering
- Better extraction and validation logic

### `factchecking_ToCF.py`
Performs fact-checking using original claims and their counterfactuals.

**Key Features:**
- Enhanced aggregation strategy
- Weighted voting system
- Confidence-based decision making
- Robust label extraction

## 🛠️ Installation

```bash
# Install required dependencies
pip install torch transformers pandas scikit-learn matplotlib seaborn tqdm

# For 4-bit quantization (recommended for memory efficiency)
pip install bitsandbytes accelerate
```

## 📊 Usage Examples

### 1. Generate Counterfactuals

```bash
# Basic usage
python generate_ToCF.py input_statements.csv counterfactuals_output.csv --token YOUR_HF_TOKEN

# With custom model
python generate_ToCF.py input_statements.csv counterfactuals_output.csv \
    --model "meta-llama/Llama-2-13b-chat-hf" \
    --token YOUR_HF_TOKEN

# Tree generation mode
python generate_ToCF.py input_statements.csv counterfactuals_output.csv \
    --token YOUR_HF_TOKEN \
    --use-tree \
    --tree-depth 2

# Test mode (first 5 statements)
python generate_ToCF.py input_statements.csv counterfactuals_output.csv \
    --token YOUR_HF_TOKEN \
    --test-mode
```

### 2. Fact-Checking with ToCF

```bash
# Basic fact-checking
python factchecking_ToCF.py counterfactuals.csv factcheck_data.csv results.csv --token YOUR_HF_TOKEN

# With custom aggregation threshold
python factchecking_ToCF.py counterfactuals.csv factcheck_data.csv results.csv \
    --token YOUR_HF_TOKEN \
    --aggregation-threshold 0.7

# Test mode
python factchecking_ToCF.py counterfactuals.csv factcheck_data.csv results.csv \
    --token YOUR_HF_TOKEN \
    --test-mode

# Evaluation only (load existing results)
python factchecking_ToCF.py counterfactuals.csv factcheck_data.csv results.csv \
    --token YOUR_HF_TOKEN \
    --evaluate-only
```

## 📋 Input Data Format

### For Counterfactual Generation (`input_statements.csv`):
```csv
Statement
"The company's revenue increased by 25% in Q3 2023."
"The new vaccine shows 95% effectiveness in clinical trials."
```

### For Fact-Checking (`factcheck_data.csv`):
```csv
Statement,Context,Evidence_List,labels
"Statement to verify","Background context","['Evidence 1', 'Evidence 2']",0
```

**Label Mapping:**
- `0`: Support (statement is supported by evidence)
- `1`: Refuted (statement is contradicted by evidence)  
- `2`: NEI (Not Enough Information)

## 📈 Output Format

### Counterfactuals Output:
```csv
Statement,counterfactual_1,counterfactual_2,counterfactual_count,status
"Original statement","Counterfactual 1","Counterfactual 2",2,"success"
```

### Fact-Checking Results:
```csv
Statement,Context,Evidence_List,labels,predicted_label,processing_status,counterfactual_1,counterfactual_2
"Statement","Context","Evidence",0,1,"success","CF1","CF2"
```

## ⚙️ Configuration Options

### Generation Parameters
- `--batch-size`: Number of statements to process in parallel (default: 4)
- `--tree-depth`: Depth for tree generation mode (default: 2)
- `--aggregation-threshold`: Confidence threshold for label aggregation (default: 0.6)

### Model Options
- `--model`: HuggingFace model name (default: Qwen/Qwen2.5-72B-Instruct)
- `--disable-cache`: Disable result caching for memory savings

## 🎯 Best Practices

### 1. **Model Selection**
- Use instruction-tuned models (Llama-2-Chat, Qwen-Instruct, etc.)
- Larger models (13B+) generally perform better
- Consider memory constraints when choosing model size

### 2. **Hyperparameter Tuning**
- **Aggregation Threshold**: 0.6-0.7 for balanced performance
- **Batch Size**: Adjust based on GPU memory (1-8)
- **Tree Depth**: 2-3 levels provide good coverage without redundancy

### 3. **Data Quality**
- Ensure statements are clear and factual
- Provide relevant context and evidence
- Balance dataset across different domains

### 4. **Performance Optimization**
- Use GPU acceleration when available
- Enable caching for repeated runs
- Consider quantization for memory efficiency

## 📊 Performance Metrics

The system outputs comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **Confusion Matrix**: Visual representation of classification results
- **Processing Statistics**: Success rates and cache usage

## 🔧 Troubleshooting

### Common Issues:

1. **Memory Errors**: Reduce batch size or use smaller model
2. **Token Limit**: Ensure HuggingFace token has model access
3. **Poor Counterfactuals**: Adjust temperature or try different model
4. **Label Extraction Fails**: Check prompt format and model responses

### Debug Mode:
```bash
# Enable verbose logging
export TRANSFORMERS_VERBOSITY=info
python factchecking_ToCF.py --test-mode ...
```

## 📚 Citation

If you use this enhanced ToCF system in your research, please cite:

```bibtex
@misc{enhanced_tocf_2024,
  title={Enhanced Tree of Counterfactual Fact-Checking System},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- New features
- Documentation updates

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.