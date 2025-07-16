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
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class CounterfactualGenerator:
    def __init__(self, model_name: str, access_token: str, enable_caching: bool = True):
        self.model_name = model_name
        self.access_token = access_token
        self.model = None
        self.tokenizer = None
        self.cache = {} if enable_caching else None
        self.initialize_model()

    def initialize_model(self):
        """Initialize model with optimized configuration"""
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=self.access_token,
                model_max_length=2048,
                padding_side="left"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.access_token,
                device_map='auto',
                quantization_config=quantization_config,
                torch_dtype=torch.float16
            )

            # Enable gradient checkpointing for memory optimization
            self.model.gradient_checkpointing_enable()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model initialized with optimizations enabled")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def _get_cache_key(self, statement: str) -> str:
        """Generate cache key for statement"""
        return hashlib.md5(statement.encode()).hexdigest()

    def generate_counterfactual_with_retry(self, statement: str, max_retries: int = 2) -> Optional[List[str]]:
        """Generate counterfactuals with smart retry logic"""
        for attempt in range(max_retries):
            try:
                return self._generate_counterfactual_attempt(statement)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
                
                time.sleep(0.5 + random.uniform(0, 0.5))
        
        return None

    def _generate_counterfactual_attempt(self, statement: str) -> Optional[List[str]]:
        """Single attempt to generate counterfactuals"""
        prompt = self._create_prompt(statement)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._extract_counterfactuals(response, statement)

    def generate_counterfactual(self, statement: str) -> Optional[List[str]]:
        """Generate counterfactuals with caching"""
        # Check cache first
        if self.cache is not None:
            cache_key = self._get_cache_key(statement)
            if cache_key in self.cache:
                return self.cache[cache_key]

        # Generate counterfactuals
        result = self.generate_counterfactual_with_retry(statement)
        
        # Cache result
        if self.cache is not None:
            cache_key = self._get_cache_key(statement)
            self.cache[cache_key] = result
        
        return result

    def generate_batch_counterfactuals(self, statements: List[str], batch_size: int = 4) -> List[Optional[List[str]]]:
        """Generate counterfactuals for multiple statements in batches"""
        results = []
        
        for i in range(0, len(statements), batch_size):
            batch = statements[i:i + batch_size]
            
            # Filter and cache check
            valid_batch = []
            batch_results = [None] * len(batch)
            
            for j, stmt in enumerate(batch):
                # Check cache
                if self.cache is not None:
                    cache_key = self._get_cache_key(stmt)
                    if cache_key in self.cache:
                        batch_results[j] = self.cache[cache_key]
                        continue
                
                valid_batch.append((j, stmt))
            
            # Process uncached statements
            if valid_batch:
                try:
                    indices, uncached_statements = zip(*valid_batch)
                    prompts = [self._create_prompt(stmt) for stmt in uncached_statements]
                    
                    # Tokenize batch
                    inputs = self.tokenizer(prompts, return_tensors="pt", 
                                          padding=True, truncation=True, max_length=1024)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.7,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            early_stopping=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                    
                    # Process results
                    for idx, (j, stmt) in enumerate(valid_batch):
                        response = self.tokenizer.decode(outputs[idx], skip_special_tokens=True)
                        cfs = self._extract_counterfactuals(response, stmt)
                        batch_results[j] = cfs
                        
                        # Cache result
                        if self.cache is not None:
                            cache_key = self._get_cache_key(stmt)
                            self.cache[cache_key] = cfs
                
                except Exception as e:
                    logger.warning(f"Batch processing failed, using fallback: {str(e)}")
                    # Fallback to individual processing
                    for j, stmt in valid_batch:
                        batch_results[j] = self.generate_counterfactual_with_retry(stmt)
            
            results.extend(batch_results)
            
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def generate_counterfactual_tree(self, statement: str, depth: int = 2) -> Optional[List[str]]:
        """Generate tree of counterfactuals with batch optimization"""
        print(f"\nGenerating tree for: \"{statement}\"\n")
        
        current_level = [statement]
        all_counterfactuals = []
        
        for level in range(depth):
            print(f"Level {level + 1}:")
            
            # Process current level in batch
            level_results = self.generate_batch_counterfactuals(current_level, batch_size=4)
            next_level = []
            
            for stmt, cfs in zip(current_level, level_results):
                print(f"Root: {stmt}")
                
                if cfs:
                    for i, cf in enumerate(cfs, 1):
                        print(f"   Counterfactual {i}: {cf}")
                        all_counterfactuals.append(cf)
                        if level + 1 < depth:
                            next_level.append(cf)
                else:
                    print("   No counterfactuals generated")
                
                print("-" * 60)
            
            if not next_level:
                break
                
            current_level = next_level[:4]  # Limit branching
        
        # Remove duplicates
        unique_cfs = list(dict.fromkeys(all_counterfactuals))
        print(f"\nGenerated {len(unique_cfs)} unique counterfactuals.")
        return unique_cfs if unique_cfs else None

    def _create_prompt(self, statement: str) -> str:
        """Create Vietnamese counterfactual generation prompt"""
        return f"""Tạo 2 phản thực mâu thuẫn với tuyên bố gốc:

GỐC: "{statement}"

Yêu cầu: Mâu thuẫn trực tiếp, hợp lý, cùng độ dài
Ví dụ - Gốc: "Doanh thu tăng 20%" 
Phản thực: "Doanh thu giảm 15%"

1. 
2. """

    def _extract_counterfactuals(self, response: str, original_statement: str) -> Optional[List[str]]:
        """Extract counterfactuals from response"""
        try:
            # Simple numbered pattern extraction
            pattern = r"(?:^|\n)\d+\.\s*(.+?)(?=\n\d+\.|\n*$)"
            matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)
            
            if not matches:
                # Fallback: look for any lines that look like statements
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                matches = [re.sub(r'^[\d\.\-•*\s]+', '', line).strip() 
                          for line in lines 
                          if len(line) > 20 and not line.startswith(('Gốc:', 'Yêu cầu:', 'Ví dụ:'))]
            
            # Clean and validate
            valid_cfs = []
            for cf in matches[:3]:
                cf = re.sub(r'\s+', ' ', cf.strip())
                if (15 < len(cf) < 500 and 
                    cf.lower() != original_statement.lower() and
                    not cf.startswith(('**', 'Gốc:', 'Yêu cầu:'))):
                    valid_cfs.append(cf)
            
            return valid_cfs[:2] if valid_cfs else None
            
        except Exception as e:
            logger.error(f"Error extracting counterfactuals: {str(e)}")
            return None
    


    def process_dataset(self, input_file: str, output_file: str, statement_column: str = "Statement", 
                       use_tree: bool = False, tree_depth: int = 2, batch_size: int = 4):
        """Process dataset with batch optimization"""
        try:
            # Load dataset
            df = pd.read_csv(input_file)
            logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
            
            # Validate column
            if statement_column not in df.columns:
                likely_cols = [col for col in df.columns if any(keyword in col.lower() 
                              for keyword in ['statement', 'text', 'content', 'sentence'])]
                if likely_cols:
                    statement_column = likely_cols[0]
                    logger.info(f"Using column '{statement_column}'")
                else:
                    raise ValueError(f"Column '{statement_column}' not found")
            
            statements = df[statement_column].tolist()
            
            # Initialize result columns
            if use_tree:
                df['tree_counterfactuals'] = None
                df['tree_counterfactual_count'] = 0
                df['tree_status'] = 'pending'
            else:
                df['counterfactual_1'] = None
                df['counterfactual_2'] = None
                df['counterfactual_count'] = 0
                df['status'] = 'pending'
            
            # Process statements
            if use_tree:
                # Tree mode - individual processing
                for idx, statement in tqdm(enumerate(statements), total=len(statements)):                    
                    result = self.generate_counterfactual_tree(str(statement), depth=tree_depth)
                    if result:
                        df.at[idx, 'tree_counterfactuals'] = json.dumps(result, ensure_ascii=False)
                        df.at[idx, 'tree_counterfactual_count'] = len(result)
                        df.at[idx, 'tree_status'] = 'success'
                    else:
                        df.at[idx, 'tree_status'] = 'failed'
            else:
                # Batch mode
                results = []
                for i in tqdm(range(0, len(statements), batch_size), desc="Processing batches"):
                    batch = statements[i:i + batch_size]
                    batch_results = self.generate_batch_counterfactuals(batch, batch_size)
                    results.extend(batch_results)
                
                # Update dataframe
                for idx, result in enumerate(results):
                    if result:
                        df.at[idx, 'counterfactual_1'] = result[0]
                        if len(result) > 1:
                            df.at[idx, 'counterfactual_2'] = result[1]
                        df.at[idx, 'counterfactual_count'] = len(result)
                        df.at[idx, 'status'] = 'success'
                    else:
                        df.at[idx, 'status'] = 'failed'
            
            # Save results
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # Generate summary
            if use_tree:
                success_count = (df['tree_status'] == 'success').sum()
                total_cfs = df['tree_counterfactual_count'].sum()
            else:
                success_count = (df['status'] == 'success').sum()
                total_cfs = df['counterfactual_count'].sum()
            
            summary = {
                'total_statements': len(df),
                'successful_generations': int(success_count),
                'success_rate': float(success_count / len(df) * 100),
                'total_counterfactuals': int(total_cfs),
                'cache_size': len(self.cache) if self.cache else 0
            }
            
            summary_file = output_file.replace('.csv', '_summary.json')
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_file}")
            logger.info(f"Success rate: {summary['success_rate']:.1f}%")
            logger.info(f"Total counterfactuals: {summary['total_counterfactuals']}")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Generate counterfactuals (Optimized)')
    parser.add_argument('input_file', help='Input CSV file')
    parser.add_argument('output_file', help='Output CSV file')
    parser.add_argument('--model', default='Qwen/Qwen2.5-72B-Instruct', help='Model name')
    parser.add_argument('--token', required=True, help='HuggingFace token')
    parser.add_argument('--statement-column', default='Statement', help='Statement column name')
    parser.add_argument('--test-mode', action='store_true', help='Test with first 5 statements')
    parser.add_argument('--use-tree', action='store_true', help='Use tree generation')
    parser.add_argument('--tree-depth', type=int, default=2, help='Tree depth')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--disable-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        generator = CounterfactualGenerator(
            args.model, 
            args.token, 
            enable_caching=not args.disable_cache
        )
        
        if args.test_mode:
            df = pd.read_csv(args.input_file)
            test_statements = df[args.statement_column].head(5).tolist()
            
            print(f"\nTEST MODE ({'TREE' if args.use_tree else 'BATCH'})")
            print("=" * 60)
            
            for i, statement in enumerate(test_statements, 1):
                print(f"\n{i}. Original: {statement}")
                
                if args.use_tree:
                    result = generator.generate_counterfactual_tree(statement, args.tree_depth)
                else:
                    result = generator.generate_counterfactual(statement)
                
                if result:
                    for j, cf in enumerate(result, 1):
                        print(f"   Counterfactual {j}: {cf}")
                else:
                    print("   No counterfactuals generated")
        else:
            generator.process_dataset(
                args.input_file,
                args.output_file,
                args.statement_column,
                use_tree=args.use_tree,
                tree_depth=args.tree_depth,
                batch_size=args.batch_size
            )
            
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()