import torch
import torch.nn as nn
import timm

class GeMPooling(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.tensor(p, dtype=torch.float32))
        else:
            self.register_buffer("p", torch.tensor(p, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] -> pooled: [B, C]
        p = torch.clamp(self.p, 1.0, 10.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=1).pow(1.0 / p)
        return x

class AdaptiveSpectralGating(nn.Module):
    """
    Adaptive Frequency-Domain Gating (SpectraViT Core).
    """
    def __init__(self, dim: int, h: int = 16, w: int = 16, dropout_p: float = 0.10, mod_scale: float = 0.15):
        super().__init__()
        self.h, self.w = h, w
        self.dropout_p = dropout_p
        self.mod_scale = mod_scale

        # Base complex weights in rFFT domain: [C, H, Wf, 2]
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)

        # CLS-conditioned modulation
        self.modulator = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # rFFT2
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        base = torch.view_as_complex(self.complex_weight).unsqueeze(0) # [1, C, H, Wf]
        
        # Modulation
        gamma = self.modulator(context).view(B, C, 1, 1)
        weight = base * (1.0 + self.mod_scale * gamma)
        
        # Frequency Dropout
        if self.training and self.dropout_p > 0:
            keep = torch.rand((B, 1, H, W // 2 + 1), device=x.device) > self.dropout_p
            weight = weight * keep
            
        x_filtered = x_fft * weight
        x_out = torch.fft.irfft2(x_filtered, s=(H, W), norm='ortho')
        return x_out

class SpectraViT(nn.Module):
    def __init__(self, 
                 num_classes: int = 9, 
                 img_size: int = 224, 
                 backbone_name: str = 'vit_small_patch14_dinov2.lvd142m',
                 spectral_dropout_p: float = 0.10,
                 unfreeze_blocks: list = None):
        super().__init__()
        
        # Backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, img_size=img_size)
        
        # Partial Finetuning
        if unfreeze_blocks:
            for name, param in self.backbone.named_parameters():
                requires_grad = False
                for pattern in unfreeze_blocks:
                    if pattern in name:
                        requires_grad = True
                        break
                param.requires_grad = requires_grad
        
        self.embed_dim = self.backbone.embed_dim
        embed_dim = self.embed_dim
        self.grid_hw = img_size // 14
        
        # ASG Module
        self.spectral_gating = AdaptiveSpectralGating(embed_dim, h=self.grid_hw, w=self.grid_hw, dropout_p=spectral_dropout_p)
        
        # Fusion Head
        self.gem = GeMPooling(p=3.0, trainable=True)
        self.spec_norm = nn.LayerNorm(embed_dim)
        self.cls_norm = nn.LayerNorm(embed_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Classifiers
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.aux_spec_classifier = nn.Linear(embed_dim, num_classes)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        
    def _forward_tokens(self, x):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        return x
        
    def forward(self, x):
        tokens = self._forward_tokens(x)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        
        B, N, C = patch_tokens.shape
        H = W = self.grid_hw
        
        # Grid reshape
        patch_grid = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # Spectral Gating
        spectral_feat = self.spectral_gating(patch_grid, cls_token)
        
        # Pooling
        spectral_flat = spectral_feat.flatten(2).transpose(1, 2)
        spectral_pool = self.gem(spectral_flat)
        
        spectral_pool = self.spec_norm(spectral_pool)
        cls_token = self.cls_norm(cls_token) # Norm before fusion usually good
        
        # Aux Logits
        aux_logits = self.aux_spec_classifier(spectral_pool)
        
        # Fusion
        g = self.gate(torch.cat([cls_token, spectral_pool], dim=1))
        fused = g * cls_token + (1.0 - g) * spectral_pool
        
        logits = self.classifier(fused)
        embeddings = self.proj_head(fused)
        
        return logits, embeddings, spectral_feat, aux_logits
