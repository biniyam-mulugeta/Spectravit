# DINOv2 Backbone Ablation

**Backbone**: `vit_small_patch14_dinov2.lvd142m`
**Patch Size**: 14
**Grid Size**: 16x16 (Input 224)
**Embedding Dim**: 384
**Pretraining**: Self-Supervised (DINOv2)

## Sanity Check
Running dummy forward pass...
- Baseline: Pass
- SpectraGate: Pass
- No approximation used: CLS token + 16x16 Patch Grid extracted directly.
