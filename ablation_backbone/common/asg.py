import torch
import torch.nn as nn

class AdaptiveSpectralGating(nn.Module):
    """
    Adaptive Frequency-Domain Gating (ASG).
    Refactored for reuse in ablation study.
    """
    def __init__(self, dim: int, h: int = 16, w: int = 16, dropout_p: float = 0.10, mod_scale: float = 0.15):
        super().__init__()
        self.h, self.w = h, w
        self.dropout_p = dropout_p
        self.mod_scale = mod_scale

        # Base complex weights in rFFT domain: [C, H, Wf, 2]
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w // 2 + 1, 2, dtype=torch.float32) * 0.02)

        # CLS-conditioned modulation (per-channel)
        self.modulator = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W] token-grid features
        context: [B, C] CLS token
        """
        B, C, H, W = x.shape
        # Ensure we are not silently failing if grid size mismatches, though we might allow dynamic if needed.
        # Strict mode:
        assert H == self.h and W == self.w, f"ASG initialized for {self.h}x{self.w} but got {H}x{W}"

        # rFFT2: [B, C, H, Wf]
        x_fft = torch.fft.rfft2(x, norm='ortho')

        base = torch.view_as_complex(self.complex_weight)  # [C, H, Wf]
        base = base.unsqueeze(0)  # [1, C, H, Wf]

        # Per-image, per-channel modulation
        gamma = self.modulator(context).view(B, C, 1, 1)
        weight = base * (1.0 + self.mod_scale * gamma)

        # Frequency dropout
        if self.training and self.dropout_p > 0:
            keep = torch.rand((B, 1, H, W // 2 + 1), device=x.device) > self.dropout_p
            weight = weight * keep

        x_filtered = x_fft * weight
        # irFFT2
        x_out = torch.fft.irfft2(x_filtered, s=(H, W), norm='ortho')
        return x_out
