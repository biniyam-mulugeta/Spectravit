import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.spectravit import SpectraViT
from src.data.dataset import CRCTileDataset, scan_folder_to_df

def get_corruption_transform(severity: int, img_size: int):
    """
    Returns a compoisition of transforms simulating corruption at given severity (1-5).
    Using standard corruptions approximation: Gaussian Blur, Noise, Color Jitter.
    0 means no corruption.
    """
    if severity == 0:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Severity definitions (approximate)
    blur_sigma = [0, 0.5, 1.0, 1.5, 2.0, 2.5][severity]
    noise_var = [0, 0.01, 0.03, 0.05, 0.08, 0.12][severity]
    jitter_s = [0, 0.1, 0.2, 0.3, 0.4, 0.5][severity]

    t_list = [transforms.ToPILImage()]
    
    # Color Jitter
    if jitter_s > 0:
        t_list.append(transforms.ColorJitter(brightness=jitter_s, contrast=jitter_s, saturation=jitter_s))
        
    t_list.append(transforms.Resize((img_size, img_size)))
    
    # Gaussian Blur
    if blur_sigma > 0:
        # Kernel size must be odd
        k = int(blur_sigma * 4) + 1
        if k % 2 == 0: k += 1
        t_list.append(transforms.GaussianBlur(kernel_size=k, sigma=blur_sigma))
        
    t_list.append(transforms.ToTensor())
    
    # Gaussian Noise (Add to Tensor)
    # We can use a lambda
    if noise_var > 0:
        t_list.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * (noise_var ** 0.5)))
        t_list.append(transforms.Lambda(lambda x: torch.clamp(x, 0, 1)))

    t_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
    return transforms.Compose(t_list)

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg['system']['device'])
    
    # Load Model
    model = SpectraViT(
        num_classes=cfg['data']['num_classes'],
        img_size=cfg['data']['img_size'],
        backbone_name=cfg['model']['backbone']
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # Data scan
    data_dir = args.data_dir if args.data_dir else cfg['data']['train_dir']
    df = scan_folder_to_df(data_dir, cfg['data']['class_map'])
    
    results = []
    
    # Loop over severities
    for severity in range(6): # 0 to 5
        print(f"Evaluating Severity {severity}...")
        
        transform = get_corruption_transform(severity, cfg['data']['img_size'])
        
        ds = CRCTileDataset(
            df, data_dir,
            transform=transform, normalize=None, # Normalize is inside transform
            do_mixup=False,
            img_size=cfg['data']['img_size']
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                logits = model(images)
                if isinstance(logits, tuple): logits = logits[0]
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"  > Acc: {acc:.4f} | F1: {f1:.4f}")
        results.append({
            "Severity": severity,
            "Accuracy": acc,
            "F1_Macro": f1
        })
        
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(args.out_dir, "robustness_results.csv"), index=False)
    print(f"Saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="robustness_output")
    args = parser.parse_args()
    main(args)
