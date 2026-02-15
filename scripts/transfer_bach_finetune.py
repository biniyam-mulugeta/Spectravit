import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.spectravit import SpectraViT
from src.data.dataset import CRCTileDataset, scan_folder_to_df
from src.data.transforms import get_transforms
from src.utils.seeding import seed_everything
from src.train.trainer import train_one_epoch, validate

def main(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['system']['seed'])
    device = torch.device(cfg['system']['device'])
    
    # BACH Map
    bach_map = {'Benign': 0, 'InSitu': 1, 'Invasive': 2, 'Normal': 3}
    num_classes = 4
    
    # Load Data
    df = scan_folder_to_df(args.data_dir, bach_map)
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_tf, val_tf, normalize = get_transforms(cfg['data']['img_size'])
    # Enable mixup for finetuning usually
    train_ds = CRCTileDataset(train_df, args.data_dir, transform=train_tf, normalize=normalize, do_mixup=True) 
    val_ds = CRCTileDataset(val_df, args.data_dir, transform=val_tf, normalize=normalize, do_mixup=False)
    
    weights = 1.0 / train_df['label'].value_counts().sort_index().values
    sample_weights = weights[train_df['label'].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=4) # Smaller BS for finetine if GPU constrained
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"Loading pretrained backbone from {args.ckpt}")
    model = SpectraViT(num_classes=9, img_size=cfg['data']['img_size'])
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    
    # Modify Head
    embed_dim = model.classifier.in_features
    model.classifier = nn.Linear(embed_dim, num_classes)
    model.aux_spec_classifier = nn.Linear(embed_dim, num_classes)
    
    # Differential Learning Rate
    # Unfreeze everything for full finetune, or keep early layers frozen
    for param in model.backbone.parameters():
        param.requires_grad = True # Full finetune of backbone often better for transfer
        
    backbone_ids = list(map(id, model.backbone.parameters()))
    backbone_params = filter(lambda p: id(p) in backbone_ids, model.parameters())
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())
    
    # Low LR for backbone, Higher for head
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 5e-6},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-2)
    
    # We use our trainer which expects SpectraLoss interface
    # but since num classes changed, we need new loss with 4 classes
    from src.losses.spectra_loss import SpectraLoss
    criterion = SpectraLoss(lambda_focal=0.5, lambda_supcon=0.5, lambda_l1=0, lambda_tv=0).to(device)
    
    model = model.to(device)
    print("Starting Finetuning...")
    
    best_f1 = 0.0
    for epoch in range(1, 21): 
        # Aux weight decay strategy or fixed small value? Fixed small usually ok for transfer.
        stats, f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, current_aux_weight=0.1
        )
        v_loss, v_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch} | T_F1: {f1:.4f} | V_F1: {v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(args.out_dir, "bach_finetune_best.pth"))
            
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="transfer_outputs")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)
