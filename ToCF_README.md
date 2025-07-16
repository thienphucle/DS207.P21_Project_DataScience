# Enhanced Tree of Counterfactual (ToCF) Fact-Checking System for Vietnamese

This repository contains an improved implementation of the Tree of Counterfactual (ToCF) approach for automated fact-checking in Vietnamese. The system generates counterfactual statements and uses them to enhance fact-checking accuracy through multi-perspective reasoning, specifically optimized for Vietnamese language and cultural context.

## 🚀 Key Improvements Made

### 1. **Enhanced Vietnamese Prompting Strategy**
- **Optimized Vietnamese Prompts**: Carefully crafted Vietnamese prompts for better model understanding
- **Structured Format**: Clear, step-by-step instructions in natural Vietnamese with explicit formatting requirements
- **Cultural Context**: Added reasoning process guidance and examples relevant to Vietnamese fact-checking scenarios

### 2. **Robust Label Extraction**
- **Multiple Patterns**: Supports various output formats ([LABEL: X], [Label: X], etc.)
- **Fallback Logic**: Multiple extraction strategies if primary pattern fails
- **Case Insensitive**: Handles different capitalization patterns

### 3. **Advanced Aggregation Strategy**
- **Weighted Voting**: Original statement gets double weight in decision making
- **Confidence Levels**: Different thresholds for high/medium/low confidence decisions
- **Consistency Checking**: Analyzes agreement between counterfactuals
- **Smart Fallback**: Defaults to NEI when high disagreement detected

### 4. **Improved Vietnamese Counterfactual Generation**
- **Vietnamese Contradiction Detection**: Recognizes Vietnamese contradictory terms and negation patterns
- **Quality Filtering**: Ensures counterfactuals are realistic and grammatically correct in Vietnamese
- **Multiple Extraction Patterns**: Robust parsing of Vietnamese generated counterfactuals
- **Cultural Relevance**: Validates that counterfactuals make sense in Vietnamese context

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
"Doanh thu của công ty tăng 25% trong quý 3 năm 2023."
"Vaccine mới cho thấy hiệu quả 95% trong thử nghiệm lâm sàng."
"Việt Nam đã xuất khẩu 6,2 triệu tấn gạo trong năm 2023."
```

### For Fact-Checking (`factcheck_data.csv`):
```csv
Statement,Context,Evidence_List,labels
"Doanh thu của công ty tăng 25% trong quý 3 năm 2023.","Công ty ABC công bố kết quả kinh doanh quý 3","['Báo cáo tài chính cho thấy doanh thu tăng 25%', 'CEO xác nhận tăng trưởng mạnh']",0
"Vaccine mới có hiệu quả 95% phòng ngừa COVID-19","Nghiên cứu về vaccine mới của hãng XYZ","['Thử nghiệm lâm sàng giai đoạn 3 cho kết quả 95%', 'WHO chưa phê duyệt vaccine này']",2
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

### 3. **Vietnamese Data Quality**
- Ensure statements are clear and factual in Vietnamese
- Provide relevant context and evidence in Vietnamese
- Balance dataset across different Vietnamese domains (politics, economy, health, etc.)
- Consider Vietnamese cultural context and terminology
- Use proper Vietnamese diacritics and formatting

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