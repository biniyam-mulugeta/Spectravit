import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import torch
import yaml
from itertools import cycle

# --- ECE (Expected Calibration Error) ---
def calculate_ece(probs, labels, n_bins=15):
    """
    Calculates the Expected Calibration Error of a model.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)
    
    ece = 0.0
    bin_lowers = np.linspace(0, 1, n_bins + 1)[:-1]
    bin_uppers = np.linspace(0, 1, n_bins + 1)[1:]

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
            
    return ece, bin_accuracies, bin_confidences, bin_counts

def plot_reliability_diagram(bin_accuracies, bin_confidences, bin_counts, ece, output_path, n_bins=15):
    """
    Plots a reliability diagram.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Reliability plot
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax1.plot(bin_confidences, bin_accuracies, marker='o', linestyle='-', color='blue', label='Model')
    ax1.set_xlabel("Average Confidence in Bin", fontsize=18)
    ax1.set_ylabel("Accuracy in Bin", fontsize=18)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Reliability Diagram (ECE = {ece:.4f})", fontsize = 22)
    ax1.legend(fontsize = 20)
    ax1.grid(True, alpha=0.3)
    
    # Bar chart for bin counts
    bin_centers = np.linspace(0, 1, n_bins + 1)[:-1] + (1/(2*n_bins))
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, edgecolor='black', color='lightblue', label='Sample Count')
    ax2.set_xlabel("Confidence", fontsize=18)
    ax2.set_ylabel("Count", fontsize=18)
    ax2.set_xlim(0, 1)
    ax2.legend(loc='upper right', fontsize = 20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved reliability diagram to {output_path}")

def main(args):
    # --- 1. Load Data ---
    print(f"Loading inference results from {args.input_npz}...")
    data = np.load(args.input_npz, allow_pickle=True)
    
    y_prob = data['probabilities']
    y_pred_indices = data['predictions']
    y_true_str = data['true_labels']
    embeddings = data.get('embeddings') # Use .get() for safety
    
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Filter out noise classes (e.g., .ipynb_checkpoints) ---
    keep_mask = np.array([not label.startswith('.') for label in y_true_str])
    if not np.all(keep_mask):
        print(f"Filtering out {np.sum(~keep_mask)} samples with invalid labels (e.g. .ipynb_checkpoints)")
        y_prob = y_prob[keep_mask]
        y_pred_indices = y_pred_indices[keep_mask]
        y_true_str = y_true_str[keep_mask]
        if embeddings is not None and len(embeddings) > 0:
            embeddings = embeddings[keep_mask]

    # --- 2. Map Labels ---
    # Create a sorted mapping from string labels to integer indices
    unique_labels = sorted(list(set(y_true_str)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    y_true_indices = np.array([label_to_idx[label] for label in y_true_str])
    num_classes = len(unique_labels)
    
    print(f"Found {num_classes} classes: {unique_labels}")

    # --- 3. Classification Report & Confusion Matrix ---
    print("\n--- Classification Metrics ---")
    report_text = classification_report(y_true_str, [idx_to_label[i] for i in y_pred_indices], labels=unique_labels)
    print(report_text)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    cm = confusion_matrix(y_true_str, [idx_to_label[i] for i in y_pred_indices], labels=unique_labels)
    #cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.title('Confusion Matrix', fontsize = 22)
    plt.xticks(rotation=45, ha='right', fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=300)
    print(f"Saved confusion matrix to {args.output_dir}")
    plt.close()

    # --- 4. ROC Curve (One-vs-Rest) ---
    print("\n--- Generating ROC Curve ---")
    y_true_bin = label_binarize(y_true_indices, classes=range(num_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {idx_to_label[i]} (area = {roc_auc[i]:.2f})')

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.title('Multi-class Receiver Operating Characteristic (ROC)', fontsize = 22)
    plt.legend(loc="lower right", fontsize = 20)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "roc_curve.png"), dpi=300)
    print(f"Saved ROC curve to {args.output_dir}")
    plt.close()

    # --- 5. t-SNE Plot ---
    if embeddings is not None and embeddings.shape[0] > 0:
        print("\n--- Generating t-SNE Plot ---")
        if len(embeddings) > args.tsne_samples:
            print(f"Subsampling {args.tsne_samples} points for t-SNE from {len(embeddings)} total.")
            indices = np.random.choice(len(embeddings), args.tsne_samples, replace=False)
            embeddings_sample = embeddings[indices]
            labels_sample = y_true_str[indices]
        else:
            embeddings_sample = embeddings
            labels_sample = y_true_str

        tsne = TSNE(n_components=2, verbose=1, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(embeddings_sample)
        
        df_tsne = pd.DataFrame(tsne_results, columns=['tsne-2d-one', 'tsne-2d-two'])
        df_tsne['label'] = labels_sample
        
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            hue="label",
            palette=sns.color_palette("tab10", num_classes),
            data=df_tsne,
            legend="full",
            alpha=0.8,
            s = 150,
            linewidth=0.5,
        )
        plt.title("t-SNE of Model Embeddings", fontsize = 22)
        plt.xlabel("", fontsize = 18)
        plt.ylabel("", fontsize = 18)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "tsne_plot.png"), bbox_inches='tight', dpi=300)
        print(f"Saved t-SNE plot to {args.output_dir}")
        plt.close()
    else:
        print("\n--- Skipping t-SNE (No embeddings found) ---")

    # --- 6. ECE Reliability Diagram ---
    print("\n--- Generating ECE Reliability Diagram ---")
    ece, bin_accs, bin_confs, bin_counts = calculate_ece(y_prob, y_true_indices)
    plot_reliability_diagram(bin_accs, bin_confs, bin_counts, ece, 
                             os.path.join(args.output_dir, "reliability_diagram.png"))
    
    print("\nAll plots and reports saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation plots from inference results.")
    parser.add_argument("--input_npz", required=True, help="Path to the .npz file from inference.py")
    parser.add_argument("--output_dir", default="evaluation_plots", help="Directory to save plots and reports.")
    parser.add_argument("--tsne_samples", type=int, default=2000, help="Number of samples for t-SNE plot for large datasets.")
    args = parser.parse_args()
    main(args)