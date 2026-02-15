import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml

# --- Add parent directory to path to import src ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.models.spectravit import SpectraViT
    from src.data.dataset import CRCTileDataset, scan_folder_to_df
except ImportError:
    print("Error: Could not import src modules. Make sure you are running this from the correct directory structure.")
    sys.exit(1)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        # Handle different dataset return formats
        if isinstance(batch, (list, tuple)):
            images = batch[0]
            labels = batch[1]
        else:
            images = batch['image']
            labels = batch['label']
            
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Handle tuple output if model returns aux outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
                labels = batch[1]
            else:
                images = batch['image']
                labels = batch['label']
            
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1

def main():
    parser = argparse.ArgumentParser(description="Transfer Learning: General (14-class) -> Specific (CRC)")
    parser.add_argument("--data_dir", required=True, help="Path to Colorectal (CRC) dataset root")
    parser.add_argument("--ckpt", required=True, help="Path to the 14-class pretrained model checkpoint")
    parser.add_argument("--output_dir", default="outputs/transfer_gen2spec", help="Output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_backbone", action='store_true', help="Freeze backbone layers (Linear Probe)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Setup (CRC)
    # Standard CRC classes
    crc_classes = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    class_map = {c: i for i, c in enumerate(crc_classes)}
    print(f"Target Classes (CRC): {crc_classes}")

    # Scan folder
    df = scan_folder_to_df(args.data_dir, class_map)
    
    # Transforms
    img_size = 224
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset & Split
    full_dataset = CRCTileDataset(df, args.data_dir, transform=train_transform, normalize=None, img_size=img_size)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2. Model Setup
    print("Initializing model with 14 classes (to match checkpoint)...")
    model = SpectraViT(num_classes=14, img_size=img_size)
    
    print(f"Loading weights from {args.ckpt}...")
    state_dict = torch.load(args.ckpt, map_location=device)
    # Clean state dict keys if needed
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    # 3. Replace Head
    print(f"Replacing head for {len(crc_classes)} CRC classes...")
    # Use classifier instead of head to match SpectraViT architecture
    in_features = model.classifier.in_features if hasattr(model.classifier, 'in_features') else model.embed_dim
    model.classifier = nn.Linear(in_features, len(crc_classes))
    # Update aux classifier as well to prevent shape mismatches
    if hasattr(model, 'aux_spec_classifier'):
        model.aux_spec_classifier = nn.Linear(in_features, len(crc_classes))
        
    model.to(device)

    if args.freeze_backbone:
        print("Freezing backbone layers...")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    # 4. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    best_acc = 0.0
    
    print("Starting Transfer Learning (General -> Specific)...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, "best_model_gen2spec.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            # Save class map for inference
            with open(os.path.join(args.output_dir, "class_map.yaml"), 'w') as f:
                yaml.dump(class_map, f)

    print(f"Training complete. Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()