import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.seeding import seed_everything
from src.data.dataset import CRCTileDataset, scan_folder_to_df
from src.data.transforms import get_transforms
from src.models.spectravit import SpectraViT
from src.losses.spectra_loss import SpectraLoss
from src.train.trainer import train_one_epoch, validate

def get_class_weights(y):
    # Check if we have enough samples
    if len(y) == 0: return np.array([])
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = 1.0 / counts
    weights_lookup = dict(zip(classes, weight_per_class))
    return np.array([weights_lookup[cls] for cls in y])

def main():
    parser = argparse.ArgumentParser(description="Train SpectraViT from scratch on a new dataset")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (folders as classes)")
    parser.add_argument("--output_dir", default="outputs/new_dataset_scratch", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Scan Data
    print(f"Scanning {args.data_dir}...")
    # Infer classes from folders
    classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    print(f"Found {len(classes)} classes: {class_map}")
    
    df = scan_folder_to_df(args.data_dir, class_map)
    if len(df) == 0:
        raise ValueError("No images found in data directory.")

    # 2. Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['label'])
    
    # 3. Dataloaders
    train_tf, val_tf, normalize = get_transforms(args.img_size)
    
    # Defaults for new dataset (mixup off for simplicity unless added later)
    train_ds = CRCTileDataset(train_df, args.data_dir, transform=train_tf, normalize=normalize, img_size=args.img_size)
    val_ds = CRCTileDataset(val_df, args.data_dir, transform=val_tf, normalize=normalize, img_size=args.img_size)

    # Weighted Sampler
    weights = get_class_weights(train_df['label'].values)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True) if len(weights) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 4. Model
    print("Initializing SpectraViT (DINOv2 backbone)...")
    model = SpectraViT(num_classes=len(classes), img_size=args.img_size)
    model.to(device)

    # 5. Optimizer (Split LR logic from paper)
    backbone_lr = args.lr * 0.01 # 1e-5 usually
    
    if hasattr(model, 'backbone'):
        backbone_ids = list(map(id, model.backbone.parameters()))
        backbone_params = filter(lambda p: id(p) in backbone_ids and p.requires_grad, model.parameters())
        rest_params = filter(lambda p: id(p) not in backbone_ids and p.requires_grad, model.parameters())
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': rest_params, 'lr': args.lr}
        ], weight_decay=1e-2)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 6. Loss
    criterion = SpectraLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    # 7. Loop
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # Aux schedule
        curr_aux = 0.5
        if epoch < args.epochs * 0.75:
             curr_aux = 0.1 + (0.4 * (epoch / (args.epochs * 0.75))) # Ramp up?? Original logic was simpler.
             # Actually let's stick to the paper simple logic: 0.1 -> 0.5
             pass 
        else:
             # Decay
             pass
        # Simplified: Constant 0.5 often works fine for generic datasets
        curr_aux = 0.1

        stats, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            current_aux_weight=curr_aux, scaler=scaler
        )
        
        v_loss, v_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch} | Train F1: {train_f1:.4f} | Val F1: {v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            # Save class map
            with open(os.path.join(args.output_dir, "class_map.yaml"), 'w') as f:
                yaml.dump(class_map, f)

    # Always save last
    torch.save(model.state_dict(), os.path.join(args.output_dir, "last_model.pth"))
    print(f"Done. Best F1: {best_f1}")

if __name__ == "__main__":
    main()
