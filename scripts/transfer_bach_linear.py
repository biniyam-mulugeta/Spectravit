import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.spectravit import SpectraViT
from src.data.dataset import CRCTileDataset, scan_folder_to_df
from src.data.transforms import get_transforms
from src.utils.seeding import seed_everything
from src.train.trainer import train_one_epoch, validate

def main(args):
    # This script assumes BACH or similar structure
    # Config is used for generic params, but we override classes for BACH
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['system']['seed'])
    device = torch.device(cfg['system']['device'])
    
    # BACH Map (Example)
    bach_map = {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3}
    num_classes = 4
    
    # Load Data
    print(f"Scanning BACH data at {args.data_dir}")
    df = scan_folder_to_df(args.data_dir, bach_map)
    
    # Split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_tf, val_tf, normalize = get_transforms(cfg['data']['img_size'])
    
    train_ds = CRCTileDataset(train_df, args.data_dir, transform=train_tf, normalize=normalize, do_mixup=False) # No mixup for linear probe usually
    val_ds = CRCTileDataset(val_df, args.data_dir, transform=val_tf, normalize=normalize, do_mixup=False)
    
    # Sampler
    class_counts = train_df['label'].value_counts().sort_index().values
    weights = 1.0 / class_counts
    sample_weights = weights[train_df['label'].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    
    # Load Pretrained SpectraViT
    print(f"Loading pretrained backbone from {args.ckpt}")
    
    # Initialize with CRC config (9 classes) to load weights safely
    model = SpectraViT(num_classes=9, img_size=cfg['data']['img_size'])
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    
    # Modify Head for BACH (4 classes)
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False
        
    # Re-init classifier
    embed_dim = model.classifier.in_features
    model.classifier = nn.Linear(embed_dim, num_classes)
    # Also fix aux classifier just in case, though unused in Linear Probe
    model.aux_spec_classifier = nn.Linear(embed_dim, num_classes)
    
    # Unfreeze only classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # Optimizer (Only Head)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Linear Probe Training...")
    
    best_acc = 0.0
    for epoch in range(1, 11): # 10 epochs for linear probe
        # Using simplified train loop (no aux weight needed for linear probe of frozen model)
        # But our trainer handles tuple output.
        # SpectraViT returns tuple. Trainer handles it.
        # Loss: We just care about main logits for linear probe.
        
        # We need a simple wrapper for criterion to pick 1st output and compute CE
        def simple_crit(logits, *args, **kwargs):
            return criterion(logits, kwargs.get('labels') if 'labels' in kwargs else args[-1])
            
        # Or better, just define a lambda
        # Our SpectraLoss expects full signature.
        # Let's just manually run a simple loop here or adapt trainer.
        # Adapting trainer: pass a custom criterion wrapper.
        
        model.train()
        losses = []
        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(img)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        # Validation
        val_acc, val_f1 = validate_simple(model, val_loader, device)
        
        print(f"Epoch {epoch} | Loss: {np.mean(losses):.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, "bach_linear_best.pth"))
            
    print("Done.")

def validate_simple(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for img, lbl in loader:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.argmax(logits, dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(lbl.cpu().numpy())
    from sklearn.metrics import accuracy_score, f1_score
    return accuracy_score(labels, preds), f1_score(labels, preds, average='macro')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="transfer_outputs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
