import argparse
import logging
import sys
import torch
import pandas as pd
import re
import json
import hashlib
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('factcheck_tocf.log')
    ]
)
logger = logging.getLogger(__name__)

class ToCFFactChecker:
    def __init__(self, model_name: str, access_token: str, enable_caching: bool = True):
        self.model_name = model_name
        self.access_token = access_token
        self.model = None
        self.tokenizer = None
        self.cache = {} if enable_caching else None
        self.label_mapping = {"Support": 0, "Refuted": 1, "NEI": 2}
        self.reverse_label_mapping = {0: "Support", 1: "Refuted", 2: "NEI"}
        self.initialize_model()

    def initialize_model(self):
        """Initialize the model and tokenizer with optimized configuration"""
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.access_token,
                model_max_length=2048,
                padding_side="left",
                truncation_side="left"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.access_token,
                device_map='auto',
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )

            self.model.gradient_checkpointing_enable()

            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

            logger.info("Model initialized successfully with optimizations enabled")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def _get_cache_key(self, statement: str, context: str = "", evidence: str = "") -> str:
        """Generate cache key for statement, context, and evidence combination"""
        combined = f"{statement}|{context}|{evidence}"
        return hashlib.md5(combined.encode()).hexdigest()

    def load_counterfactuals_from_file(self, input_file: str, statement_column: str = "Statement") -> pd.DataFrame:
        """Load counterfactuals from the output of generate_ToCF.py"""
        try:
            df = pd.read_csv(input_file)
            logger.info(f"Loaded counterfactuals dataset with {len(df)} rows and columns: {list(df.columns)}")
            
            # Check if required columns exist
            required_columns = [statement_column]
            counterfactual_columns = []
            
            # Check for different counterfactual column formats
            if 'counterfactual_1' in df.columns:
                counterfactual_columns = ['counterfactual_1', 'counterfactual_2']
            elif 'tree_counterfactuals' in df.columns:
                counterfactual_columns = ['tree_counterfactuals']
            else:
                # Look for any column containing 'counterfactual'
                counterfactual_columns = [col for col in df.columns if 'counterfactual' in col.lower()]
            
            if not counterfactual_columns:
                raise ValueError("No counterfactual columns found in the input file")
            
            logger.info(f"Found counterfactual columns: {counterfactual_columns}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading counterfactuals file: {str(e)}")
            raise

    def extract_counterfactuals_from_row(self, row: pd.Series) -> List[str]:
        """Extract counterfactuals from a dataframe row"""
        counterfactuals = []
        
        # Handle tree counterfactuals (JSON format)
        if 'tree_counterfactuals' in row and pd.notna(row['tree_counterfactuals']):
            try:
                tree_cfs = json.loads(row['tree_counterfactuals'])
                if isinstance(tree_cfs, list):
                    counterfactuals.extend(tree_cfs)
            except json.JSONDecodeError:
                logger.warning("Failed to parse tree_counterfactuals JSON")
        
        # Handle individual counterfactual columns
        for i in range(1, 10):  # Check up to 10 counterfactual columns
            col_name = f'counterfactual_{i}'
            if col_name in row and pd.notna(row[col_name]):
                counterfactuals.append(str(row[col_name]))
        
        # Remove duplicates while preserving order
        unique_counterfactuals = list(dict.fromkeys(counterfactuals))
        return [cf for cf in unique_counterfactuals if cf and len(cf.strip()) > 5]

    def parse_evidence(self, evidence_data: Any) -> List[str]:
        """Parse evidence data handling both strings and lists"""
        if pd.isna(evidence_data):
            return []
        
        # If it's already a list, return it
        if isinstance(evidence_data, list):
            return [str(ev) for ev in evidence_data if ev]
        
        # If it's a string, try to parse it
        evidence_str = str(evidence_data).strip()
        if not evidence_str:
            return []
        
        # Try to parse as JSON list first
        try:
            parsed = json.loads(evidence_str)
            if isinstance(parsed, list):
                return [str(ev) for ev in parsed if ev]
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try to parse as Python list using eval (be careful with this)
        try:
            parsed = eval(evidence_str)
            if isinstance(parsed, list):
                return [str(ev) for ev in parsed if ev]
        except (SyntaxError, NameError, TypeError):
            pass
        
        # If all parsing fails, treat as single evidence string
        return [evidence_str]

    def generate_fact_check_prompt(self, statement: str, context: str, evidence_list: List[str], 
                                 counterfactuals: List[str]) -> str:
        """Generate optimized Vietnamese prompt for fact-checking using ToCF approach"""
        evidence_text = "\n".join([f"• {ev}" for ev in evidence_list]) if evidence_list else "Không có bằng chứng được cung cấp."
        counterfactual_text = "\n".join([f"• {cf}" for cf in counterfactuals]) if counterfactuals else "Không có phản thực được cung cấp."

        prompt = f"""Bạn là chuyên gia kiểm tra sự thật. Nhiệm vụ của bạn là xác minh tính chính xác của các tuyên bố sử dụng bằng chứng và lý luận phản thực.

**TUYÊN BỐ CẦN XÁC MINH:**
"{statement}"

**BỐI CẢNH:**
{context if context else "Không có bối cảnh bổ sung."}

**BẰNG CHỨNG HỖ TRỢ:**
{evidence_text}

**CÁC TUYÊN BỐ PHẢN THỰC:**
{counterfactual_text}

**HƯỚNG DẪN:**
1. Phân tích tuyên bố dựa trên bằng chứng được cung cấp
2. Xem xét cách các tuyên bố phản thực giúp tiết lộ tính xác thực của tuyên bố
3. Phân loại tuyên bố vào CHÍNH XÁC MỘT danh mục:
   - Support: Bằng chứng trực tiếp ủng hộ tuyên bố
   - Refuted: Bằng chứng trực tiếp bác bỏ tuyên bố
   - NEI: Không đủ thông tin để xác định tính xác thực

**QUY TRÌNH LẬP LUẬN:**
- So sánh tuyên bố với bằng chứng
- Đánh giá tính nhất quán với các sự kiện đã biết
- Xem xét các kịch bản thay thế từ phản thực
- Đưa ra quyết định cuối cùng

**ĐỊNH DẠNG ĐẦU RA:** Chỉ xuất ra nhãn theo định dạng chính xác này:
[LABEL: Support] HOẶC [LABEL: Refuted] HOẶC [LABEL: NEI]

**QUY TẮC QUAN TRỌNG:**
- CHỈ trả lời bằng một trong ba nhãn: Support, Refuted, NEI
- KHÔNG thêm giải thích hay văn bản khác
- Đảm bảo định dạng chính xác: [LABEL: X]

[LABEL:"""
        return prompt

    def stance_detection(self, statement: str, context: str, evidence_list: List[str], 
                        counterfactuals: List[str], max_retries: int = 3) -> int:
        """Detect stance for a single statement with counterfactuals"""
        # Check cache first
        if self.cache is not None:
            cache_key = self._get_cache_key(statement, context, str(evidence_list + counterfactuals))
            if cache_key in self.cache:
                return self.cache[cache_key]

        for attempt in range(max_retries):
            try:
                prompt = self.generate_fact_check_prompt(statement, context, evidence_list, counterfactuals)
                
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1024, 
                    padding="max_length"
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # Extract label from response with multiple patterns
                response_label = "NEI"  # Default fallback
                
                # Try multiple extraction patterns
                patterns = [
                    r"\[LABEL:\s*(Support|Refuted|NEI)\]",
                    r"\[Label:\s*(Support|Refuted|NEI)\]", 
                    r"(?:LABEL|Label):\s*(Support|Refuted|NEI)",
                    r"\b(Support|Refuted|NEI)\b(?:\s*[\]\)])?$"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        response_label = match.group(1).title()  # Ensure proper case
                        break
                
                # Final fallback: look for label keywords
                if response_label == "NEI":
                    response_lower = response.lower()
                    if "support" in response_lower and "refuted" not in response_lower:
                        response_label = "Support"
                    elif "refuted" in response_lower and "support" not in response_lower:
                        response_label = "Refuted"
                
                result = self.label_mapping[response_label]
                
                # Cache result
                if self.cache is not None:
                    cache_key = self._get_cache_key(statement, context, str(evidence_list + counterfactuals))
                    self.cache[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.warning(f"Stance detection attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    return 2  # Default to NEI on failure
                
                time.sleep(0.5 + random.uniform(0, 0.5))
        
        return 2  # Default to NEI

    def evaluate_stance_with_counterfactuals(self, statement: str, context: str, evidence_list: List[str], 
                                           counterfactuals: List[str], aggregation_threshold: float = 0.6) -> int:
        """Evaluate stance using counterfactuals with enhanced aggregation"""
        if not counterfactuals:
            # If no counterfactuals, use original statement only
            return self.stance_detection(statement, context, evidence_list, [])
        
        # Get stance for original statement (weighted more heavily)
        original_stance = self.stance_detection(statement, context, evidence_list, [])
        
        # Get stances for each counterfactual  
        counterfactual_stances = []
        for cf in counterfactuals[:5]:  # Limit to first 5 counterfactuals for efficiency
            cf_stance = self.stance_detection(cf, context, evidence_list, [])
            counterfactual_stances.append(cf_stance)
        
        # Enhanced aggregation strategy
        if not counterfactual_stances:
            return original_stance
            
        # Weighted voting: original statement gets double weight
        weighted_votes = [original_stance, original_stance] + counterfactual_stances
        label_counts = Counter(weighted_votes)
        total = len(weighted_votes)
        
        # Calculate ratios
        support_ratio = label_counts[0] / total if total > 0 else 0
        refuted_ratio = label_counts[1] / total if total > 0 else 0
        nei_ratio = label_counts[2] / total if total > 0 else 0
        
        # Confidence-based decision making
        max_ratio = max(support_ratio, refuted_ratio, nei_ratio)
        
        # High confidence threshold
        if max_ratio >= 0.7:
            if support_ratio == max_ratio:
                return 0  # Support
            elif refuted_ratio == max_ratio:
                return 1  # Refuted
            else:
                return 2  # NEI
        
        # Medium confidence - prefer original stance unless contradicted
        elif max_ratio >= aggregation_threshold:
            # If original stance agrees with majority, prefer it
            if original_stance == label_counts.most_common(1)[0][0]:
                return original_stance
            else:
                return label_counts.most_common(1)[0][0]
        
        # Low confidence - use consistency check
        else:
            # Check if counterfactuals are consistent
            cf_counter = Counter(counterfactual_stances)
            cf_consistency = len(cf_counter) <= 2  # Most agree on 1-2 labels
            
            if cf_consistency and original_stance in cf_counter:
                return original_stance
            else:
                # Default to NEI when there's high disagreement
                return 2  # NEI

    def process_fact_checking_dataset(self, counterfactuals_file: str, fact_check_file: str, 
                                    output_file: str, statement_column: str = "Statement",
                                    context_column: str = "Context", evidence_column: str = "Evidence_List",
                                    label_column: str = "labels", batch_size: int = 1,
                                    aggregation_threshold: float = 0.6):
        """Process fact-checking dataset using pre-generated counterfactuals"""
        try:
            # Load counterfactuals
            cf_df = self.load_counterfactuals_from_file(counterfactuals_file, statement_column)
            
            # Load fact-checking dataset
            fc_df = pd.read_csv(fact_check_file)
            logger.info(f"Loaded fact-checking dataset with {len(fc_df)} rows")
            
            # Validate columns
            required_columns = [statement_column, context_column, evidence_column]
            missing_columns = [col for col in required_columns if col not in fc_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in fact-checking file: {missing_columns}")
            
            # Merge datasets on statement column
            merged_df = fc_df.merge(cf_df, on=statement_column, how='left', suffixes=('', '_cf'))
            logger.info(f"Merged dataset has {len(merged_df)} rows")
            
            # Initialize results
            predicted_labels = []
            processing_status = []
            
            # Process each row
            for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing fact-checking"):
                try:
                    statement = str(row[statement_column])
                    context = str(row[context_column]) if pd.notna(row[context_column]) else ""
                    
                    # Parse evidence list using improved parsing
                    evidence_list = self.parse_evidence(row[evidence_column])
                    
                    # Extract counterfactuals
                    counterfactuals = self.extract_counterfactuals_from_row(row)
                    
                    # Perform fact-checking (removed validate_statement check)
                    predicted_label = self.evaluate_stance_with_counterfactuals(
                        statement, context, evidence_list, counterfactuals, aggregation_threshold
                    )
                    
                    predicted_labels.append(predicted_label)
                    processing_status.append('success')
                    
                    logger.info(f"Processed row {idx}: Statement='{statement[:50]}...', "
                              f"Evidence count={len(evidence_list)}, Counterfactuals={len(counterfactuals)}, "
                              f"Predicted={self.reverse_label_mapping[predicted_label]}")
                    
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {str(e)}")
                    predicted_labels.append(2)  # Default to NEI on error
                    processing_status.append('error')
                
                # Clear GPU cache periodically
                if torch.cuda.is_available() and (idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Add results to dataframe
            merged_df['predicted_label'] = predicted_labels
            merged_df['processing_status'] = processing_status
            
            # Save results
            merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # Generate summary
            summary = self.generate_processing_summary(merged_df, label_column)
            
            # Save summary
            summary_file = output_file.replace('.csv', '_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Fact-checking results saved to: {output_file}")
            logger.info(f"Summary saved to: {summary_file}")
            
            return merged_df, summary
            
        except Exception as e:
            logger.error(f"Error processing fact-checking dataset: {str(e)}")
            raise

    def generate_processing_summary(self, df: pd.DataFrame, label_column: str = "labels") -> Dict[str, Any]:
        """Generate summary statistics for fact-checking results"""
        summary = {
            'total_statements': len(df),
            'processing_status': df['processing_status'].value_counts().to_dict(),
            'predicted_label_distribution': df['predicted_label'].value_counts().to_dict(),
            'cache_size': len(self.cache) if self.cache else 0
        }
        
        # Add evaluation metrics if true labels are available
        if label_column in df.columns:
            valid_rows = df[df['processing_status'] == 'success']
            if len(valid_rows) > 0:
                y_true = valid_rows[label_column].tolist()
                y_pred = valid_rows['predicted_label'].tolist()
                
                summary['evaluation_metrics'] = {
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
                }
                
                # Per-class metrics
                for i, label_name in self.reverse_label_mapping.items():
                    if i in y_true:
                        precision = precision_score(y_true, y_pred, labels=[i], average=None, zero_division=0)
                        recall = recall_score(y_true, y_pred, labels=[i], average=None, zero_division=0)
                        f1 = f1_score(y_true, y_pred, labels=[i], average=None, zero_division=0)
                        
                        summary['evaluation_metrics'][f'{label_name.lower()}_precision'] = float(precision[0]) if len(precision) > 0 else 0.0
                        summary['evaluation_metrics'][f'{label_name.lower()}_recall'] = float(recall[0]) if len(recall) > 0 else 0.0
                        summary['evaluation_metrics'][f'{label_name.lower()}_f1'] = float(f1[0]) if len(f1) > 0 else 0.0
        
        return summary

    def evaluate_model_performance(self, results_df: pd.DataFrame, label_column: str = "labels", 
                                 save_plots: bool = True, output_dir: str = "."):
        """Evaluate model performance and generate visualizations"""
        try:
            valid_rows = results_df[results_df['processing_status'] == 'success']
            
            if len(valid_rows) == 0:
                logger.warning("No valid rows found for evaluation")
                return
            
            y_true = valid_rows[label_column].tolist()
            y_pred = valid_rows['predicted_label'].tolist()
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            
            print(f"\n{'='*50}")
            print(f"FACT-CHECKING PERFORMANCE EVALUATION")
            print(f"{'='*50}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"{'='*50}")
            
            # Generate confusion matrix
            if save_plots:
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                           xticklabels=["Support", "Refuted", "NEI"], 
                           yticklabels=["Support", "Refuted", "NEI"])
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title("ToCF Fact-Checking Confusion Matrix")
                plt.tight_layout()
                
                plot_path = Path(output_dir) / "confusion_matrix.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.show()
                
                logger.info(f"Confusion matrix saved to: {plot_path}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {str(e)}")
            return None

def main():
    parser = argparse.ArgumentParser(description='ToCF-based Fact-Checking System')
    parser.add_argument('counterfactuals_file', help='CSV file with pre-generated counterfactuals')
    parser.add_argument('fact_check_file', help='CSV file with statements to fact-check')
    parser.add_argument('output_file', help='Output CSV file for results')
    parser.add_argument('--model', default='ura-hcmut/ura-llama-13b', help='Model name')
    parser.add_argument('--token', required=True, help='HuggingFace access token')
    parser.add_argument('--statement-column', default='Statement', help='Statement column name')
    parser.add_argument('--context-column', default='Context', help='Context column name')
    parser.add_argument('--evidence-column', default='Evidence_List', help='Evidence column name')
    parser.add_argument('--label-column', default='labels', help='True label column name')
    parser.add_argument('--aggregation-threshold', type=float, default=0.6, 
                       help='Threshold for label aggregation (default: 0.6)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--disable-cache', action='store_true', help='Disable result caching')
    parser.add_argument('--test-mode', action='store_true', help='Test with first 5 statements')
    parser.add_argument('--evaluate-only', action='store_true', 
                       help='Only evaluate existing results (skip processing)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.counterfactuals_file).exists():
        logger.error(f"Counterfactuals file not found: {args.counterfactuals_file}")
        sys.exit(1)
    
    if not Path(args.fact_check_file).exists():
        logger.error(f"Fact-check file not found: {args.fact_check_file}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Initializing ToCF Fact-Checker with model: {args.model}")
        fact_checker = ToCFFactChecker(
            args.model, 
            args.token, 
            enable_caching=not args.disable_cache
        )
        
        if args.evaluate_only:
            # Load existing results and evaluate
            if not Path(args.output_file).exists():
                logger.error(f"Results file not found for evaluation: {args.output_file}")
                sys.exit(1)
            
            results_df = pd.read_csv(args.output_file)
            fact_checker.evaluate_model_performance(results_df, args.label_column)
        else:
            # Process fact-checking dataset
            if args.test_mode:
                logger.info("Running in test mode with first 5 statements...")
                # Load and limit datasets for testing
                cf_df = pd.read_csv(args.counterfactuals_file).head(5)
                fc_df = pd.read_csv(args.fact_check_file).head(5)
                
                cf_test_file = args.counterfactuals_file.replace('.csv', '_test.csv')
                fc_test_file = args.fact_check_file.replace('.csv', '_test.csv')
                
                cf_df.to_csv(cf_test_file, index=False)
                fc_df.to_csv(fc_test_file, index=False)
                
                results_df, summary = fact_checker.process_fact_checking_dataset(
                    cf_test_file,
                    fc_test_file,
                    args.output_file,
                    args.statement_column,
                    args.context_column,
                    args.evidence_column,
                    args.label_column,
                    args.batch_size,
                    args.aggregation_threshold
                )
                
                # Clean up test files
                Path(cf_test_file).unlink()
                Path(fc_test_file).unlink()
            else:
                logger.info("Processing full fact-checking dataset...")
                results_df, summary = fact_checker.process_fact_checking_dataset(
                    args.counterfactuals_file,
                    args.fact_check_file,
                    args.output_file,
                    args.statement_column,
                    args.context_column,
                    args.evidence_column,
                    args.label_column,
                    args.batch_size,
                    args.aggregation_threshold
                )
            
            # Evaluate performance
            if args.label_column in results_df.columns:
                fact_checker.evaluate_model_performance(results_df, args.label_column)
            
            logger.info("Fact-checking completed successfully!")
            
    except Exception as e:
        logger.error(f"Fact-checking failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()