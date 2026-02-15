import argparse
import yaml
import torch
import torch.nn as nn
import os
import sys
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.spectravit import SpectraViT
from src.data.transforms import get_transforms
from src.data.dataset import scan_folder_to_df

# Local
from ablation_utils import create_radial_mask, get_frequency_bands
from plotting import plot_dominance_curve, plot_importance_heatmap

class AblationASGWrapper(nn.Module):
    """
    Wraps an existing ASG module to inject a frequency mask.
    Does NOT copy weights; just forwards to original instance with mask logic.
    """
    def __init__(self, original_asg):
        super().__init__()
        self.original_asg = original_asg
        self.mask = None # If None, identity pass
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Original logic reused, but with mask injection
        # We REPLICATE forward logic here because we need to intervene mid-stream
        
        B, C, H, W = x.shape
        # rFFT2
        x_fft = torch.fft.rfft2(x, norm='ortho') # [B, C, H, Wf]
        
        # --- ABLATION POINT ---
        if self.mask is not None:
            # Mask is [1, 1, H, Wf] usually
            # Zero out masked frequencies? 
            # "Either remove (zero) or keep only". Usually ablation means remove.
            # Let's assume Mask=1 means REMOVE (ablate). 
            # Or Mask=1 means KEEP?
            # Standard ablation: "Remove Band X" -> Mask=0 for Band X, 1 elsewhere.
            # Our util returns mask=1 for the band.
            # So we should MULTIPLY by (1 - mask) to remove that band.
            x_fft = x_fft * (1.0 - self.mask)
        # ----------------------
        
        # Continue with original weights
        base = torch.view_as_complex(self.original_asg.complex_weight).unsqueeze(0)
        gamma = self.original_asg.modulator(context).view(B, C, 1, 1)
        weight = base * (1.0 + self.original_asg.mod_scale * gamma)
        
        # Dropout (skip for analysis usually, model.eval handles it)
        
        x_filtered = x_fft * weight
        x_out = torch.fft.irfft2(x_filtered, s=(H, W), norm='ortho')
        return x_out

def run_ablation(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print(f"Loading checkpoint: {cfg['paths']['ckpt_path']}")
    # We need to know src config to init model? Assuming generic config or defaults
    # For now, generic.
    model = SpectraViT(num_classes=9, img_size=224, backbone_name='vit_small_patch14_dinov2.lvd142m')
    state = torch.load(cfg['paths']['ckpt_path'], map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    
    # 2. Inject Wrapper
    print("Injecting Ablation Wrapper into ASG...")
    model.spectral_gating = AblationASGWrapper(model.spectral_gating)
    
    # 3. Data Preparation
    print(f"Scanning data: {cfg['paths']['data_dir']}")
    # Using generic class map for now (CRC-100K)
    class_map = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
    df = scan_folder_to_df(cfg['paths']['data_dir'], class_map)
    
    # Stratified Sampling
    n_per_class = cfg['experiment']['samples_per_class']
    sampled_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min(len(x), n_per_class)))
    
    print(f"Running on {len(sampled_df)} total samples.")
    
    # Transform
    base_tf, _, norm = get_transforms(224)
    # We use validation transform (resize+norm)
    clean_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm
    ])
    
    # Corrupted Transform (Gaussian Blur)
    severity = cfg['experiment']['corruption']['severity'] 
    # Approx blur sigma for severity 5 ~ 2.5
    blur_sigma = [0, 0.5, 1.0, 1.5, 2.0, 2.5][severity]
    corrupt_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.GaussianBlur(kernel_size=9, sigma=blur_sigma),
        transforms.ToTensor(),
        norm
    ])
    
    # 4. Analysis Loop
    conditions = {'clean': clean_tf, 'corrupted': corrupt_tf}
    results = []
    
    bands = get_frequency_bands(cfg['experiment']['num_bands'])
    
    # For grid size (hardcoded to 16x16 for DINOv2 small)
    H, W = 16, 16 
    
    for cond_name, tf in conditions.items():
        print(f"\nCondition: {cond_name}")
        
        # Pre-load images to memory for speed? Or DataLoader.
        # Simple loop
        dataset = []
        for idx, row in sampled_df.iterrows():
            full_path = os.path.join(cfg['paths']['data_dir'], row['path'])
            img = Image.open(full_path).convert('RGB')
            # label unused for logit calculation, but used for identifying target Class
            dataset.append((tf(img), row['label']))
            
        loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['experiment']['batch_size'])
        
        # 4.1 Baseline (No Ablation)
        model.spectral_gating.mask = None
        base_logits_list = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                logits = model(imgs) 
                if isinstance(logits, tuple): logits = logits[0]
                
                # Gather logit of the TARGET CLASS specifically
                # Or max logit? User said "target-class logit".
                # We assume ground truth is available and correct.
                # Gather: logits[range(B), lbls]
                targets = logits[range(len(lbls)), lbls.to(device)]
                base_logits_list.append(targets.cpu().numpy())
                
        base_logits = np.concatenate(base_logits_list)
        
        # 4.2 Bands
        for b_idx, (r_min, r_max) in enumerate(bands):
            print(f"  Band {b_idx}: {r_min:.2f}-{r_max:.2f}")
            
            # Create Mask
            mask = create_radial_mask(H, W, r_min, r_max, device)
            model.spectral_gating.mask = mask
            
            ablated_logits_list = []
            with torch.no_grad():
                for imgs, lbls in loader:
                    imgs = imgs.to(device)
                    logits = model(imgs)
                    if isinstance(logits, tuple): logits = logits[0]
                    targets = logits[range(len(lbls)), lbls.to(device)]
                    ablated_logits_list.append(targets.cpu().numpy())
            
            ablated_logits = np.concatenate(ablated_logits_list)
            
            # Compute Drops
            drops = base_logits - ablated_logits
            
            # Record
            for d in drops:
                results.append({
                    "condition": cond_name,
                    "band_idx": b_idx,
                    "r_min": r_min,
                    "logit_drop": d
                })
                
    # 5. Save & Plot
    out_dir = cfg['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(out_dir, "raw_drops.csv"), index=False)
    
    print("Generating Plots...")
    plot_dominance_curve(res_df, out_dir)
    plot_importance_heatmap(res_df, H, W, out_dir)
    
    print("Done.")

if __name__ == "__main__":
    # If generic config passed
    cfg_file = os.path.join(os.path.dirname(__file__), "config.yaml")
    run_ablation(cfg_file)
