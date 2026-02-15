# Controlled Backbone Ablation Study

This folder contains the implementation for demonstrating that SpectraViT's performance is architecture-driven, not backbone-driven.

## Summary of Experiments

| Backbone | Variant | Patch Size | Grid Size (224px) | Pretraining | Embed Dim |
|----------|---------|------------|-------------------|-------------|-----------|
| **DINOv2** | `dinov2` | 14 | 16x16 | Self-Supervised | 384 |
| **DeiT-Small** | `deit_small` | 16 | 14x14 | Distilled Supervised | 384 |
| **ViT-Small** | `vit_small` | 16 | 14x14 | Supervised ImageNet | 384 |
| **MAE-ViT** | `mae_vit_small` | 16 | 14x14 | Masked Autoencoder | 384 |

**Controlled Variables:**
- All models use **identical components**: `common/asg.py`, `common/gated_fusion.py`.
- All models output: `logits, embeddings, spectral_feat, aux_logits`.
- **Constraint**: Only the Grid Size varies (16x16 vs 14x14) due to the native patch size of the pre-trained backbones.

## How to Run

For each folder (`dinov2`, `deit_small`, `vit_small`, `mae_vit_small`):

```bash
cd ablation_backbone/<backbone>
python inference.py --model spectragate --ckpt path/to/ckpt.pth --data /path/to/images/
```

## Sanity Checks
Run `python run_sanity_check.py` to verify shapes and forward passes for all 8 models (4 backbones x 2 variants).
