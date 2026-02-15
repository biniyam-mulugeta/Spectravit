import argparse
import os
import sys
import pandas as pd

# Add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.data.dataset import scan_folder_to_df

def main():
    parser = argparse.ArgumentParser(description="Generate ground truth annotations CSV from a dataset folder.")
    parser.add_argument("--data_dir", required=True, help="Path to the dataset root (e.g., CRC-VAL-HE-7K)")
    parser.add_argument("--output", default="annotations.csv", help="Output CSV path")
    args = parser.parse_args()

    # Standard CRC-9 Class Map
    class_map = {
        'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 
        'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8
    }

    print(f"Scanning {args.data_dir}...")
    df = scan_folder_to_df(args.data_dir, class_map)
    
    if len(df) == 0:
        print("No images found! Check path and ensure folder structure matches class names (ADI, BACK, etc.)")
        return

    # Sort for consistency
    df = df.sort_values('path')
    
    # Save
    print(f"Saving {len(df)} annotations to {args.output}")
    df.to_csv(args.output, index=False)
    print("Done.")
    print("Format: path, label, class_name")
    print(f"Example: {df.iloc[0]['path']}, {df.iloc[0]['label']}, {df.iloc[0]['class_name']}")

if __name__ == "__main__":
    main()
