import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
import yaml
import io
from PIL import Image

# --- Add parent path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.models.spectravit import SpectraViT
from src.data.dataset import CRCTileDataset, scan_folder_to_df

# --- Corruptions ---
class Corruptions:
    @staticmethod
    def impulse_noise(img_t, severity):
        """
        Applies Impulse (Salt-and-Pepper-like) noise to a tensor image.
        severity 1-5 maps to probability of noise.
        """
        if severity == 0:
            return img_t
            
        # Severity maps to probability of a pixel being corrupted
        # p ranges from 0.01 to 0.05 roughly
        p = [0, 0.01, 0.02, 0.03, 0.04, 0.05][severity]
        
        # Create a mask for salt (1) and pepper (0)
        # We'll just flip pixels to 0 or 1.
        # Note: input img_t is likely normalized, but for noise injection it's often better on [0,1].
        # However, the user snippet applied noise to the normalized tensor. 
        # Impulse noise typically replaces pixel values.
        # Since we are in tensor land (likely normalized), "1" might be out of distribution if we don't unnormalize.
        # But to stick to the user's pattern "Simple workaround: apply noise to normalized tensor",
        # we will add noise to the tensor directly.
        
        # For impulse noise on normalized tensors, replacing with min/max of the tensor or 0/1 is tricky.
        # Let's assume we replace with random extreme values or just add shifts.
        # Actually, "Salt and Pepper" on normalized data:
        # We can create a mask.
        
        mask = torch.rand_like(img_t)
        
        # Salt (max-ish value) - let's say we boost values
        # Pepper (min-ish value) - let's say we zero values (or min)
        
        # But to be safe and simple on normalized tensors:
        # We will just use the "Impulse Noise" implementation where we add random large spikes?
        # Or strictly Salt and Pepper:
        # Salt: Probability p/2 -> set to max
        # Pepper: Probability p/2 -> set to min
        
        # Getting min/max from the batch/image itself or assuming standard norm dist?
        # Let's assume standard norm approx range [-2, 2] usually.
        
        # A safer "Impulse Noise" for tensors:
        # Add sparse strong noise.
        
        noise = torch.randn_like(img_t)
        # Only keep noise where mask < p
        mask_noise = (mask < p).float()
        
        # The noise itself should be "impulsive" i.e. large magnitude?
        # Or we can just use the provided logic: "use another noise instead of Gaussian noise".
        # Let's use Salt-and-Pepper probability based.
        
        # Copy image
        out = img_t.clone()
        
        # Salt mode (set to distinct high value, e.g. 2.0 which is ~2 std devs)
        # Pepper mode (set to distinct low value, e.g. -2.0)
        
        n_elements = img_t.numel()
        
        # Generate random indices
        mask_salt = (torch.rand_like(img_t) < (p / 2))
        mask_pepper = (torch.rand_like(img_t) < (p / 2))
        
        # We need to know 'max' and 'min'. Since it's normalized, we can't know exact 'white'/'black'.
        # But usually max value in normalized ImageNet is around >2, min < -2.
        # We will use local max/min to be safe or hardcoded values.
        # Let's use hardcoded "strong" values.
        
        out[mask_salt] = 2.0 # Bright
        out[mask_pepper] = -2.0 # Dark
        
        return out

    @staticmethod
    def gaussian_blur(img_t, severity):
        if severity == 0: return img_t
        # severity 1-5 maps to sigma
        kernel_size = [1, 3, 5, 7, 9, 11][severity]
        sigma = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5][severity]
        return transforms.GaussianBlur(kernel_size, sigma)(img_t)

    @staticmethod
    def brightness_contrast(img_t, severity):
        if severity == 0: return img_t
        # severity determines strength
        # Only modify brightness and contrast, NO hue/saturation
        s = [0, 0.1, 0.2, 0.3, 0.4, 0.5][severity]
        return transforms.ColorJitter(brightness=s, contrast=s, saturation=0, hue=0)(img_t)

    @staticmethod
    def elastic_transform(img_t, severity):
        if severity == 0: return img_t
        # Severity maps to alpha (magnitude of displacement)
        # sigma (smoothness)
        # Based on common corruption benchmarks or tissue folding simulation
        alpha = [0, 20.0, 40.0, 60.0, 80.0, 100.0][severity]
        sigma = [0, 3.0, 4.0, 5.0, 6.0, 7.0][severity]
        
        # NOTE: ElasticTransform expects tensor or PIL. 
        # Since img_t is a tensor (CHW), this works.
        # Fill=0 (black/mean in norm space)
        et = transforms.ElasticTransform(alpha=alpha, sigma=sigma)
        return et(img_t)

    @staticmethod
    def jpeg_compression(img_t, severity):
        if severity == 0: return img_t
        # Severity maps to quality (lower is worse)
        quality = [100, 90, 75, 60, 45, 30][severity]
        
        # JPEG requires uint8 [0, 255]
        # We need to Denormalize -> Quantize -> JPEG -> Normalize
        
        # Mean/Std for ImageNet (used in base_transform)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_t.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_t.device)
        
        # Denormalize
        img_un = img_t * std + mean
        img_un = torch.clamp(img_un, 0, 1)
        
        # To PIL
        img_pil = transforms.ToPILImage()(img_un)
        
        # JPEG via BytesIO
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer)
        
        # Back into Tensor and Normalize
        t_jpeg = transforms.ToTensor()(img_jpeg).to(img_t.device)
        t_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t_jpeg)
        
        return t_norm

def get_corrupted_loader(dataset, transform_fn, batch_size, num_workers):
    # Wrap dataset to apply extra transform
    class CorruptedWrapper(Dataset):
        def __init__(self, base_ds, fn):
            self.ds = base_ds
            self.fn = fn
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            # CRCTileDataset returns (img, label) strictly? Check usages.
            # Looking at robustness_eval.py, it seems it returns images, labels
            # But user snippet was img, label, path.
            # We will handle whatever the dataset returns.
            
            data = self.ds[idx]
            img_t = data[0]
            other = data[1:]
            
            # Apply noise
            img_c = self.fn(img_t)
            
            return (img_c, *other)

    wrapper = CorruptedWrapper(dataset, transform_fn)
    return DataLoader(wrapper, batch_size=batch_size, num_workers=num_workers)

def run_robustness_test(args):
    print(f"--- Robustness Testing ---\nDataset: {args.data_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Config if exists to get model params
    # We will assume some defaults if not provided in a config file, or use args
    # But usually models need specific init args.
    # The existing code loads from config.
    
    # Let's try to load the config associated with the checkpoint or default
    config_path = "configs/default.yaml"
    if os.path.exists("configs/default.yaml"):
        with open("configs/default.yaml", 'r') as f:
            cfg = yaml.safe_load(f)
            num_classes = cfg['data']['num_classes']
            img_size = cfg['data']['img_size']
            backbone = cfg['model']['backbone']
    else:
        # Fallback values
        print("Warning: Config not found, using defaults.")
        num_classes = 9 # Guessing from CRC
        img_size = 224
        backbone = 'vit_base_patch16_224'

    print(f"Model: {backbone}, Classes: {num_classes}, Size: {img_size}")

    model = SpectraViT(num_classes=num_classes, img_size=img_size, backbone_name=backbone)
    if os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print(f"Checkpoint not found: {args.ckpt}")
        return

    model.to(device)
    model.eval()

    # Load Base Dataset
    # We need a DataFrame first
    # scan_folder_to_df(root_dir, class_map)
    # We need class map.
    if 'class_map' in cfg['data']:
        class_map = cfg['data']['class_map']
    else:
        # Try to infer or default
        class_map = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
    
    df = scan_folder_to_df(args.data_dir, class_map)
    
    # Base transform (Resize + Normalize)
    # Note: CRCTileDataset applies transform. 
    # But we want to apply corruption ON TOP of the base transform (tensor)?
    # User's snippet: "apply noise to normalized tensor"
    # So base_dataset should return normalized tensors.
    
    # We need to construct the base dataset
    # We will use the standard transform logic but maybe exposure it
    
    # For simplicity, we define the base transform here
    # Standard ImageNet normalization
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    # We pass 'normalize=None' to CRCTileDataset because we included it in transform
    base_dataset = CRCTileDataset(df, args.data_dir, transform=base_transform, normalize=None, img_size=img_size)

    # Subsample for speed
    if len(base_dataset) > args.max_samples:
        indices = np.random.choice(len(base_dataset), args.max_samples, replace=False)
        base_dataset = Subset(base_dataset, indices)

    results = []

    # Test Loop: Corruption Type -> Severity 0-5
    corruptions = {
        "Impulse Noise": Corruptions.impulse_noise,
        "Gaussian Blur": Corruptions.gaussian_blur,
        "Brightness/Contrast": Corruptions.brightness_contrast,
        "Elastic Deformation": Corruptions.elastic_transform,
        "JPEG Compression": Corruptions.jpeg_compression
    }

    batch_size = 32 # or from args

    for name, fn in corruptions.items():
        print(f"\nTesting {name}...")
        for sev in range(6): # 0 = clean, 1-5 = corrupted
            # Create lambda for this specific corruption
            this_corr = lambda x: fn(x, sev)
            loader = get_corrupted_loader(base_dataset, this_corr, batch_size, num_workers=4)
            
            # Eval
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in loader:
                    imgs = batch[0].to(device)
                    lbls = batch[1] # Assumes label is second
                    
                    logits = model(imgs)
                    if isinstance(logits, tuple): logits = logits[0]
                    
                    preds = torch.argmax(logits, dim=1)
                    y_true.extend(lbls.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            print(f"  Severity {sev}: Acc = {acc:.4f} | F1 = {f1:.4f}")
            results.append({"corruption": name, "severity": sev, "accuracy": acc, "f1": f1})

    # Save and Plot
    os.makedirs(args.out_dir, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(args.out_dir, "robustness_results_impulse.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_res, x='severity', y='f1', hue='corruption', marker='o')
    plt.title("Model Robustness to Synthetic Corruptions (Macro F1)",fontsize=22)
    plt.xlabel("Corruption Severity",fontsize=22)
    plt.ylabel("Macro F1 Score",fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Corruption', fontsize='18', title_fontsize='18')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "robustness_plot_impulse_New.png"))
    print(f"Saved results and plot to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out_dir", default="ablation_extra/results")
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()
    
    run_robustness_test(args)
