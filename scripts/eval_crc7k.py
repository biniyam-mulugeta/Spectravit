import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.spectravit import SpectraViT
from src.data.dataset import CRCTileDataset, scan_folder_to_df
from src.data.transforms import get_transforms

def evaluate(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg['system']['device'])
    
    # 1. Model
    print(f"Loading model from {args.ckpt}")
    model = SpectraViT(
        num_classes=cfg['data']['num_classes'],
        img_size=cfg['data']['img_size'],
        backbone_name=cfg['model']['backbone']
    ).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # 2. Data
    data_dir = args.data_dir if args.data_dir else cfg['data']['train_dir'] # Default fallback
    print(f"Evaluating on {data_dir}")
    
    # Scan logic
    df = scan_folder_to_df(data_dir, cfg['data']['class_map'])
    
    _, val_tf, normalize = get_transforms(cfg['data']['img_size'])
    
    ds = CRCTileDataset(
        df, data_dir, 
        transform=val_tf, normalize=normalize, 
        do_mixup=False, 
        img_size=cfg['data']['img_size']
    )
    
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)
    
    # 3. Inference
    all_preds, all_labels = [], []
    all_probs = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            output = model(images)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # 4. Metrics
    # Reverse class map
    idx_to_class = {v: k for k, v in cfg['data']['class_map'].items()}
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # Save Report
    os.makedirs(args.out_dir, exist_ok=True)
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(args.out_dir, "classification_report.csv"))
    
    # Save CM
    cm = confusion_matrix(all_labels, all_preds)
    np.savetxt(os.path.join(args.out_dir, "confusion_matrix.csv"), cm, delimiter=",")
    
    print(f"Results saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", help="Path to eval dataset (folder of folders)")
    parser.add_argument("--out_dir", default="eval_outputs")
    args = parser.parse_args()
    evaluate(args)
