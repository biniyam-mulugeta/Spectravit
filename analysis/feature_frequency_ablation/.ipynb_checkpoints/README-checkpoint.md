# Feature-Frequency Ablation Study

This module allows analyzing which frequency components within the ASG spectral branch are most important for the model's predictions.

## Methodology
1.  **Radial Band Masking**: We divide the frequency spectrum (in the rFFT domain of the patch grid) into concentric rings (bands).
2.  **Ablation**: For each band, we zero out the coefficients in that ring *after* the FFT but *before* the ASG gating weights are applied.
3.  **Metric**: We measure the **drop in the target-class logit**. A large positive drop indicates the ablated features were supporting the correct prediction (high importance). A negative drop (logit increase) implies the features were noise or distractors.

## Usage

1.  **Update Config**: Edit `config.yaml` to point to your trained `.pth` checkpoint and validation dataset.
2.  **Run Analysis**:
    ```bash
    python run_ablation.py
    ```
3.  **Outputs**:
    - `outputs/raw_drops.csv`: Raw data (logit drops per sample per band).
    - `outputs/plot_A_dominance_curve.png`: Mean importance vs frequency band.
    - `outputs/plot_B_heatmap.png`: 2D reconstruction of frequency importance.

## Interpretation
- **Dominance Curve**: If the curve is high on the left (low index), the model relies on Low Frequencies (shapes, structures). If high on the right, it relies on High Frequencies (texture, fine detail).
- **Robustness**: Compare the 'Clean' vs 'Corrupted' curves. A robust model should maintain reliance on structural (low) features even when high frequencies are corrupted.
