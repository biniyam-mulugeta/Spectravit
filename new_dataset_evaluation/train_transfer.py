import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.seeding import seed_everything
from src.data.dataset import CRCTileDataset, scan_folder_to_df
from src.data.transforms import get_transforms
from src.models.spectravit import SpectraViT
from src.losses.spectra_loss import SpectraLoss
from src.train.trainer import train_one_epoch, validate

def get_class_weights(y):
    if len(y) == 0: return np.array([])
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = 1.0 / counts
    weights_lookup = dict(zip(classes, weight_per_class))
    return np.array([weights_lookup[cls] for cls in y])

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SpectraViT on a new dataset")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root")
    parser.add_argument("--ckpt", required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--output_dir", default="outputs/new_dataset_transfer", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", action='store_true', help="Freeze backbone layers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Scan Data
    print(f"Scanning {args.data_dir}...")
    classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {class_map}")
    
    df = scan_folder_to_df(args.data_dir, class_map)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['label'])
    
    train_tf, val_tf, normalize = get_transforms(args.img_size)
    train_ds = CRCTileDataset(train_df, args.data_dir, transform=train_tf, normalize=normalize, img_size=args.img_size)
    val_ds = CRCTileDataset(val_df, args.data_dir, transform=val_tf, normalize=normalize, img_size=args.img_size)

    weights = get_class_weights(train_df['label'].values)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True) if len(weights) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2. Load Model
    # Determine old num_classes from checkpoint? Hard to know. 
    # Usually standard is 9 (CRC). 
    # We initialize with a "dummy" generic 9, load weights, then replace head.
    print("Loading pretrained SpectraViT...")
    model = SpectraViT(num_classes=9, img_size=args.img_size) # Checkpoint likely trained on CRC-9
    
    # Load Weights
    try:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state, strict=False) # Strict=False to ignore head mismatch just in case
        print("Weights loaded.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # 3. Modify Head
    print(f"Replacing head for {num_classes} classes...")
    model.classifier = nn.Linear(model.embed_dim, num_classes)
    model.aux_spec_classifier = nn.Linear(model.embed_dim, num_classes)
    model.to(device)

    # 4. Freeze Logic
    if args.freeze_backbone:
        print("Freezing backbone and ASG...")
        for name, param in model.named_parameters():
             # Keep classifier/fusion/aux unfrozen
             if "classifier" in name or "fusion" in name:
                 param.requires_grad = True
             else:
                 param.requires_grad = False
    
    # 5. Optimizer
    # If frozen, only optimize filtered params
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = SpectraLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    # 6. Loop
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        stats, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            current_aux_weight=0.1, scaler=scaler
        )
        v_loss, v_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch} | Train F1: {train_f1:.4f} | Val F1: {v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            with open(os.path.join(args.output_dir, "class_map.yaml"), 'w') as f:
                yaml.dump(class_map, f)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pth"))
    print(f"Done. Best F1: {best_f1}")

if __name__ == "__main__":
    main()
