import torch
import torch.nn as nn
import timm
import sys
import os

# Add common to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../common')))

from asg import AdaptiveSpectralGating
from gated_fusion import GatedFusion, GeMPooling

class SpectraGateModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        # 1. Backbone: DINOv2 ViT Small Patch14
        # Grid = 224 / 14 = 16x16
        self.backbone_name = "vit_small_patch14_dinov2.lvd142m"
        self.backbone = timm.create_model(self.backbone_name, pretrained=True, img_size=224)
        
        self.embed_dim = self.backbone.embed_dim
        self.grid_hw = 224 // 14 # 16

        # 2. Components from Common
        self.asg = AdaptiveSpectralGating(self.embed_dim, h=self.grid_hw, w=self.grid_hw)
        self.gem_pool = GeMPooling()
        
        # Norms
        self.spec_norm = nn.LayerNorm(self.embed_dim)
        self.cls_norm = nn.LayerNorm(self.embed_dim)
        
        # Fusion
        self.fusion = GatedFusion(self.embed_dim)
        
        # Heads
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.aux_classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        # Extract tokens manually to ensure CLS + Grid
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        
        B, N, C = patch_tokens.shape
        H = W = self.grid_hw
        patch_grid = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # ASG
        spectral_feat = self.asg(patch_grid, cls_token)
        
        # Pool
        spec_flat = spectral_feat.chunk(1, dim=2)[0] # Just flatten, chunk was mistake in thought
        spec_flat = spectral_feat.flatten(2).transpose(1, 2)
        spec_pool = self.gem_pool(spec_flat)
        
        # Norm
        spec_pool = self.spec_norm(spec_pool)
        cls_token = self.cls_norm(cls_token) # Paper applied norm before fuse
        
        # Fuse
        fused = self.fusion(cls_token, spec_pool)
        
        # Logits
        main_logits = self.classifier(fused)
        aux_logits = self.aux_classifier(spec_pool)
        
        return main_logits, fused, aux_logits
