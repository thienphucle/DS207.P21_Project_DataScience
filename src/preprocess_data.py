import pandas as pd
import argparse
from pathlib import Path

def process_vfactcheck(input_file: str, output_file: str):
    """Process VFactCheck dataset."""
    try:
        df = pd.read_csv(input_file)
        
        processed_df = pd.DataFrame({
            'Statement': df['Statement'],
            'Labels': df['Labels']
        })
        
        # Save to CSV
        processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Processed VFactCheck data saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing VFactCheck dataset: {str(e)}")
        raise

def process_viwikifc(input_file: str, output_file: str):
    """Process ViWikiFC dataset."""
    try:
        df = pd.read_csv(input_file)

        label_mapping = {
            'Supports': 0,
            'Refutes': 1,
            'Not_Enough_Information': 2
        }
        df['gold_label'] = df['gold_label'].map(label_mapping)

        processed_df = pd.DataFrame({
            'Statement': df['claim'],
            'Labels': df['gold_label'],
            'Context': df['context'],
            'Evidence': df['evidence']
        })
        
        # Save to CSV
        processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Processed ViWikiFC data saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing ViWikiFC dataset: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets for counterfactual generation')
    parser.add_argument('--dataset', type=str, choices=['vfactcheck', 'viwikifc'], required=True,
                      help='Which dataset to process (vfactcheck or viwikifc)')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    
    args = parser.parse_args()
    
    if args.dataset == 'vfactcheck':
        process_vfactcheck(args.input, args.output)
    else:
        process_viwikifc(args.input, args.output)

if __name__ == "__main__":
    main()