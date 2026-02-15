import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path
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
    parser = argparse.ArgumentParser(description="Train and Evaluate SpectraViT with 60/20/20 Split")
    parser.add_argument("--data_dir", required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", default="outputs/new_dataset_split_60_20_20", help="Where to save model and results")
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
    classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    class_map = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_map.items()}
    print(f"Found {len(classes)} classes: {class_map}")
    
    df = scan_folder_to_df(args.data_dir, class_map)
    if len(df) == 0:
        raise ValueError("No images found in data directory.")

    # 2. 60/20/20 Split
    print("Performing 60/20/20 Split...")
    # First split: 60% Train, 40% Temp (Val+Test)
    train_df, temp_df = train_test_split(df, test_size=0.4, random_state=args.seed, stratify=df['label'])
    # Second split: Split Temp equally (20% Val, 20% Test)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed, stratify=temp_df['label'])
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Save splits for reproducibility
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    # 3. Dataloaders
    train_tf, val_tf, normalize = get_transforms(args.img_size)
    
    train_ds = CRCTileDataset(train_df, args.data_dir, transform=train_tf, normalize=normalize, img_size=args.img_size)
    val_ds = CRCTileDataset(val_df, args.data_dir, transform=val_tf, normalize=normalize, img_size=args.img_size)
    test_ds = CRCTileDataset(test_df, args.data_dir, transform=val_tf, normalize=normalize, img_size=args.img_size)

    weights = get_class_weights(train_df['label'].values)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True) if len(weights) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 4. Model
    print("Initializing SpectraViT...")
    model = SpectraViT(num_classes=len(classes), img_size=args.img_size)
    model.to(device)

    # 5. Optimizer
    # Split LR 
    backbone_lr = args.lr * 0.01 
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
    criterion = SpectraLoss().to(device)
    scaler = torch.amp.GradScaler('cuda')

    # 6. Training Loop
    print("\n=== Starting Training ===")
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        curr_aux = 0.5 # Constant aux weight
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

    print(f"\nTraining Complete. Best Val F1: {best_f1:.4f}")
    
    # 7. Final Evaluation on Test Set
    print("\n=== Running Final Test ====")
    # Load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            if isinstance(logits, tuple): logits = logits[0]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)
    
    with open(os.path.join(args.output_dir, "test_report.txt"), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n")
        f.write(report)
        
    # Save Predictions
    results_df = test_df.copy()
    results_df['prediction'] = [idx_to_class[p] for p in all_preds]
    results_df['correct'] = results_df['class_name'] == results_df['prediction']
    results_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
