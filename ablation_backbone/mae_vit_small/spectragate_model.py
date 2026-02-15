import torch
import torch.nn as nn
import timm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../common')))

from asg import AdaptiveSpectralGating
from gated_fusion import GatedFusion, GeMPooling

class SpectraGateModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        print("DEBUG: Loading MAE SpectraGateModel (Fixed 4-Output Version)")
        self.backbone_name = "vit_small_patch16_224"
        self.backbone = timm.create_model(self.backbone_name, pretrained=False) 
        
        self.embed_dim = self.backbone.embed_dim
        self.grid_hw = 224 // 16 # 14

        self.asg = AdaptiveSpectralGating(self.embed_dim, h=self.grid_hw, w=self.grid_hw)
        self.gem_pool = GeMPooling()
        
        self.spec_norm = nn.LayerNorm(self.embed_dim)
        self.cls_norm = nn.LayerNorm(self.embed_dim)
        self.fusion = GatedFusion(self.embed_dim)
        
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        self.aux_classifier = nn.Linear(self.embed_dim, num_classes)
        
        # Add Projection Head
        self.proj_head = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        
        B, N, C = patch_tokens.shape
        H = W = self.grid_hw
        patch_grid = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        spectral_feat = self.asg(patch_grid, cls_token)
        
        spec_flat = spectral_feat.flatten(2).transpose(1, 2)
        spec_pool = self.gem_pool(spec_flat)
        
        spec_pool = self.spec_norm(spec_pool)
        cls_token = self.cls_norm(cls_token) # Paper applied norm before fuse
        
        fused = self.fusion(cls_token, spec_pool)
        
        main_logits = self.classifier(fused)
        aux_logits = self.aux_classifier(spec_pool)
        embeddings = self.proj_head(fused)
        
        return main_logits, embeddings, spectral_feat, aux_logits
