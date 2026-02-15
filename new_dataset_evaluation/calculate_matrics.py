import os
import argparse
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Plotting libraries not available ({e}). Confusion matrix plot will be skipped.")
    PLOTTING_AVAILABLE = False
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import yaml

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics from inference CSV")
    parser.add_argument("--csv", required=True, help="Path to predictions.csv")
    parser.add_argument("--output_dir", default="evaluation_results", help="Where to save report/plots")
    parser.add_argument("--class_map", default=None, help="Path to class_map.yaml for sorting (optional)")
    parser.add_argument("--img_exts", nargs="+", default=['.png', '.jpg', '.jpeg', '.tif', '.tiff'], help="Extensions to ignore when parsing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    print(f"Loading {args.csv}...")
    df = pd.read_csv(args.csv)
    
    if 'file' not in df.columns or 'prediction_label' not in df.columns:
        raise ValueError("CSV must contain 'file' and 'prediction_label' columns.")

    # 2. Extract Ground Truth
    # Assumption: File path is .../class_name/image_name.ext
    # We take the parent folder name as the true label.
    
    def get_parent_folder(path):
        # Handle mixed slashes just in case
        path = path.replace('\\', '/')
        return os.path.basename(os.path.dirname(path))

    df['true_label'] = df['file'].apply(get_parent_folder)
    
    # Filter out .ipynb_checkpoints noise
    df = df[~df['true_label'].str.contains('ipynb_checkpoints', case=False, na=False)]

    # Handle mismatch: Predictions (Int) vs Ground Truth (Str)
    if pd.api.types.is_integer_dtype(df['prediction_label']) and not pd.api.types.is_integer_dtype(df['true_label']):
        print("\n[Info] Detected integer predictions vs string ground truth.")
        print("Mapping predicted indices to sorted ground truth classes...")
        unique_true_sorted = sorted(df['true_label'].unique())
        mapping = {i: label for i, label in enumerate(unique_true_sorted)}
        print(f"Inferred Mapping: {mapping}")
        df['prediction_label'] = df['prediction_label'].map(mapping)
        
        if df['prediction_label'].isna().any():
            df = df.dropna(subset=['prediction_label'])

    # 3. Validation
    # Check if true_labels look like random ids or actual names.
    unique_true = sorted(df['true_label'].unique())
    unique_pred = sorted(df['prediction_label'].unique())
    
    print(f"Found {len(unique_true)} ground truth classes: {unique_true}")
    print(f"Found {len(unique_pred)} predicted classes: {unique_pred}")
    
    # Optional: Align with class_map if provided
    labels_order = None
    if args.class_map and os.path.exists(args.class_map):
        with open(args.class_map, 'r') as f:
            class_map = yaml.safe_load(f)
            # class_map is Name -> Index. We just want the Names sorted by Index.
            sorted_items = sorted(class_map.items(), key=lambda x: x[1])
            labels_order = [k for k, v in sorted_items]
            print(f"Using class order from map: {labels_order}")
    else:
        # Union of all seen labels sorted
        labels_order = sorted(list(set(unique_true) | set(unique_pred)))

    # 4. Metrics
    y_true = df['true_label']
    y_pred = df['prediction_label']
    
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=labels_order, labels=labels_order, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=labels_order, labels=labels_order)
    print("\nClassification Report:")
    print(report_text)
    
    # Save Report
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report_text)
        
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels_order)
    
    if PLOTTING_AVAILABLE:
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=labels_order, yticklabels=labels_order)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Normalized Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
        print(f"Saved plots and report to {args.output_dir}")
    else:
        print(f"Saved report to {args.output_dir} (Plots skipped due to missing libraries)")

if __name__ == "__main__":
    main()
