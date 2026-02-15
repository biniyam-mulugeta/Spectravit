import torch
import numpy as np

def create_radial_mask(h: int, w: int, r_min: float, r_max: float, device: torch.device):
    """
    Creates a binary mask for the rFFT domain.
    h, w: Spatial dimensions.
    r_min, r_max: Normalized frequency radii [0.0, 1.0]. 
                  0.0 = DC component (center).
                  1.0 = Nyquist frequency (corners).
    """
    # rFFT dimensions: [H, W//2 + 1]
    # Frequencies are usually centered in standard FFT, but rfft has DC at (0,0).
    # We need to compute distance from (0,0) and alias frequencies for the H dimension.
    
    # Grid coordinates
    yc, xc = h // 2, w // 2 # Center for full FFT
    
    # We'll generate indices for the full FFT then slice for rFFT to handle wrapping correctly
    # Or cleaner: explicitly compute freq coords
    
    # PyTorch fftfreq: [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)
    # We treat 'd' as 1.
    
    fy = torch.fft.fftfreq(h, d=1.0, device=device) # [-0.5, 0.5)
    fx = torch.fft.rfftfreq(w, d=1.0, device=device) # [0, 0.5]
    
    # Meshgrid
    Y, X = torch.meshgrid(fy, fx, indexing='ij')
    
    # Radius map (Euclidean distance from DC)
    # Normalize so 0.5 (Nyquist) becomes 1.0 logic for user convenience?
    # User asked for bands. Let's assume r in [0, 0.5*sqrt(2)].
    # Let's normalize by 0.5 so max freq along axis is 1.0.
    R = torch.sqrt(Y**2 + X**2) / 0.5
    
    # Mask
    mask = (R >= r_min) & (R < r_max)
    
    # Add dimensions for broadcasting: [1, 1, H, Wf] (Batch, Channel, H, Wf)
    return mask.float().unsqueeze(0).unsqueeze(0)

def get_frequency_bands(num_bands=5):
    """
    Returns list of (r_min, r_max) tuples splitting [0, 1.41]
    Max theoretical radius in square is sqrt(1^2 + 1^2) = 1.414.
    """
    max_r = 1.414
    step = max_r / num_bands
    bands = []
    for i in range(num_bands):
        bands.append((i * step, (i + 1) * step))
    return bands
