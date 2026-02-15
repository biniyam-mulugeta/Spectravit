import os
import glob
import pandas as pd
import argparse

def generate_robustness_latex(csv_path):
    """Reads robustness_results.csv and prints a LaTeX table body."""
    try:
        df = pd.read_csv(csv_path)
        print(r"% --- Paste this into your Robustness Table ---")
        print(r"% Source: " + csv_path)
        print(r"\begin{tabular}{lcc}")
        print(r"    \toprule")
        print(r"    \textbf{Severity} & \textbf{Accuracy} & \textbf{Macro F1} \\")
        print(r"    \midrule")
        
        for _, row in df.iterrows():
            sev = int(row['Severity'])
            acc = row['Accuracy']
            f1 = row['F1_Macro']
            # Convert to percentage if in 0-1 range
            if acc <= 1.0: acc *= 100
            if f1 <= 1.0: f1 *= 100
            
            print(f"    {sev} & {acc:.2f}\% & {f1:.2f}\% \\\\")
            
        print(r"    \bottomrule")
        print(r"\end{tabular}")
        print("\n")
    except Exception as e:
        print(f"% Error parsing robustness CSV: {e}")

def generate_ablation_summary(csv_path):
    """Reads raw_drops.csv from frequency ablation and summarizes it."""
    try:
        df = pd.read_csv(csv_path)
        # Expected columns from analysis/feature_frequency_ablation: 'band_index', 'logit_drop'
        if 'band_index' in df.columns and 'logit_drop' in df.columns:
            summary = df.groupby('band_index')['logit_drop'].mean().reset_index()
            
            print(r"% --- Frequency Ablation Summary (Dominance Curve) ---")
            print(r"% Source: " + csv_path)
            print(r"% Use these values to describe the Dominance Curve or plot it.")
            print(f"{'Band':<10} | {'Mean Logit Drop':<15}")
            print("-" * 30)
            for _, row in summary.iterrows():
                print(f"{int(row['band_index']):<10} | {row['logit_drop']:.4f}")
            print("\n")
    except Exception as e:
        print(f"% Error parsing ablation CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate Results from Outputs")
    parser.add_argument("--root", default="outputs", help="Root directory to scan")
    parser.add_argument("--scan_all", action="store_true", default=True, help="Scan all default output folders")
    args = parser.parse_args()
    
    # Define directories to scan
    scan_dirs = [args.root]
    if args.scan_all:
        scan_dirs.extend(["robustness_output", "eval_outputs", "inference_outputs", "analysis/feature_frequency_ablation/outputs"])
    
    # Remove duplicates and non-existent dirs
    scan_dirs = list(set([d for d in scan_dirs if os.path.exists(d)]))
    
    print(f"% Scanning directories: {scan_dirs}...\n")
    
    found_any = False
    
    # 1. Robustness
    for d in scan_dirs:
        rob_files = glob.glob(os.path.join(d, "**", "robustness_results.csv"), recursive=True)
        if rob_files:
            generate_robustness_latex(rob_files[0])
            found_any = True
            
        ablation_files = glob.glob(os.path.join(d, "**", "raw_drops.csv"), recursive=True)
        if ablation_files:
            generate_ablation_summary(ablation_files[0])
            found_any = True

    if not found_any:
        print("% No result files found.")
        print("% 1. To generate robustness results, run: python scripts/robustness_eval.py --ckpt <path> --data_dir <path>")
        print("% 2. To generate ablation results, run: python analysis/feature_frequency_ablation/run_ablation.py")

if __name__ == "__main__":
    main()