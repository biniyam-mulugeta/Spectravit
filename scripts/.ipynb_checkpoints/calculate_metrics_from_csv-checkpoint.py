import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics by matching predictions CSV and Ground Truth CSV.")
    parser.add_argument("--pred", required=True, help="Path to predictions.csv (must have 'file' and 'prediction')")
    parser.add_argument("--gt", required=True, help="Path to annotations.csv (must have 'path' and 'label')")
    args = parser.parse_args()

    print(f"Loading predictions: {args.pred}")
    df_pred = pd.read_csv(args.pred)
    
    print(f"Loading ground truth: {args.gt}")
    df_gt = pd.read_csv(args.gt)

    # Normalize paths for matching
    # Infer prediction format: usually absolute or relative.
    # We try to strict match on the end of the string if they don't match directly.
    
    # Option A: Create a mapping from filename to label from GT
    # Assuming filenames are unique across classes (usually true for CRC dataset like 'ADI-TCGA....tif')
    
    # Helper to extract basename
    df_gt['basename'] = df_gt['path'].apply(lambda p: os.path.basename(p))
    df_pred['basename'] = df_pred['file'].apply(lambda p: os.path.basename(p))
    
    # Join
    merged = pd.merge(df_pred, df_gt, on='basename', suffixes=('_pred', '_gt'))
    
    if len(merged) == 0:
        print("ERROR: No matching files found between predictions and ground truth!")
        print("Head Pred:", df_pred['basename'].head().tolist())
        print("Head GT:", df_gt['basename'].head().tolist())
        return

    print(f"Matched {len(merged)} images.")
    
    y_true = merged['label']
    y_pred = merged['prediction']
    
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

import os
if __name__ == "__main__":
    main()
