import torch
import torch.nn as nn

class GeMPooling(nn.Module):
    """Generalized Mean Pooling over tokens."""
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

class GatedFusion(nn.Module):
    """
    Fusion Head: Gates between CLS token and Spectral Feature Pool.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, cls_token: torch.Tensor, spectral_pool: torch.Tensor) -> torch.Tensor:
        """
        cls_token: [B, C]
        spectral_pool: [B, C]
        Returns: fused [B, C]
        """
        g = self.gate_net(torch.cat([cls_token, spectral_pool], dim=1))
        fused = g * cls_token + (1.0 - g) * spectral_pool
        return fused
