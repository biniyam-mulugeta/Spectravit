import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

def plot_dominance_curve(results_df, output_dir):
    """
    Plot A: Frequency dominance curve.
    x-axis: Band Index
    y-axis: Mean Logit Drop
    """
    plt.figure(figsize=(10, 6))
    
    # Aggregation
    sns.lineplot(data=results_df, x="band_idx", y="logit_drop", hue="condition", marker="o", errorbar='sd')
    
    plt.title("Frequency Dominance Curve (Ablation Sensitivity)", fontsize=22)
    plt.xlabel("Frequency Band (Low -> High)", fontsize=18)
    plt.ylabel("Logit Drop (Importance)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Condition', fontsize='18', title_fontsize='18')
    plt.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, "plot_A_dominance_curve_New.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")

def plot_importance_heatmap(results_df, h, w, output_dir):
    """
    Plot B: Reconstruct 2D heatmap from band importance.
    Maps mean drop per band back to the spatial frequency domain.
    """
    # Create empty rFFT grid
    # rFFT size: [H, W//2 + 1]
    H, Wf = h, w // 2 + 1
    heatmap = np.zeros((H, Wf))
    
    # Get mean importance per band from CLEAN data
    clean_stats = results_df[results_df['condition'] == 'clean'].groupby('band_idx')['logit_drop'].mean()
    
    # We need to recreate the masks to fill the heatmap
    # This imports the util locally to avoid circular dep if organized poorly, 
    # but here we assume caller handles it or we re-implement simplified logic
    # Better to pass bands or reuse logic. simple radial logic:
    
    yc, xc = H // 2, 0 # rFFT DC at 0,0 for X, center for Y
    y, x = np.ogrid[:H, :Wf]
    
    # Normalize Y to [-0.5, 0.5], X to [0, 0.5]
    fy = (y - H//2) / H
    fx = x / w # x goes 0..W//2. Max freq is 0.5. x/(W)??
    # Wait, PyTorch logic: rfftfreq(w) -> [0, 1/(w), ..., 0.5]
    fx = x / w 
    
    R = np.sqrt(fy**2 + fx**2) / 0.5
    
    # Bands
    max_r = 1.414
    step = max_r / len(clean_stats)
    
    for band_idx, drop_val in clean_stats.items():
        r_min = band_idx * step
        r_max = (band_idx + 1) * step
        mask = (R >= r_min) & (R < r_max)
        heatmap[mask] = drop_val
        
    # Log scale for viz? Or lineal drop? Linear is fine for "Importance"
    plt.figure(figsize=(8, 8))
    # We typically visualize FFT with DC in center. shift logic needed for full FFT.
    # For rFFT, it's a half-plane. Let's mirror it for pretty viz.
    
    # Mirroring to create full [H, W] spectrum
    full_map = np.zeros((H, w))
    # Fill right half
    full_map[:, :Wf] = heatmap # Note: rFFT mapping is tricky.
    # Let's just plot the rFFT magnitude directly as matrix
    
    plt.imshow(heatmap, cmap='viridis', aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label("Mean Logit Drop", size=18)
    cbar.ax.tick_params(labelsize=18)
    plt.title("2D Frequency Importance (rFFT Domain)", fontsize=22)
    plt.xlabel("X Frequency (0 to Nyquist)", fontsize=18)
    plt.ylabel("Y Frequency (-Nyq to +Nyq)", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    path = os.path.join(output_dir, "plot_B_heatmap_New.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved {path}")