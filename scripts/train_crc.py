import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.seeding import seed_everything
from src.data.dataset import CRCTileDataset, scan_folder_to_df
from src.data.transforms import get_transforms
from src.models.spectravit import SpectraViT
from src.losses.spectra_loss import SpectraLoss
from src.train.trainer import train_one_epoch, validate

def get_class_weights(y):
    # Determine sample weights for WeightedRandomSampler
    # Counts
    classes, counts = np.unique(y, return_counts=True)
    # Weight = 1 / count
    weight_per_class = 1.0 / counts
    # Map back to samples
    weights_lookup = dict(zip(classes, weight_per_class))
    sample_weights = np.array([weights_lookup[cls] for cls in y])
    return sample_weights

def main(args):
    # 1. Load Config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    seed_everything(cfg['system']['seed'])
    device = torch.device(cfg['system']['device'])
    
    # 2. Data
    print(f"Loading data from {cfg['data']['train_dir']}")
    df = scan_folder_to_df(cfg['data']['train_dir'], cfg['data']['class_map'])
    
    # Stratified Split handled manually or via sklearn logic. 
    # For reproducibility, we rely on seed.
    from sklearn.model_selection import train_test_split
    val_size = 0.2
    train_df, val_df = train_test_split(
        df, test_size=val_size, 
        random_state=cfg['system']['seed'], 
        stratify=df['label']
    )
    
    train_tf, val_tf, normalize = get_transforms(cfg['data']['img_size'])
    
    # Train Dataset (Mixup enabled)
    mixup_cfg = {
        "prob": cfg['data']['mixup_prob'],
        "beta": cfg['data']['mixup_beta'],
        "strategy": cfg['data']['mixup_strategy']
    }
    
    train_ds = CRCTileDataset(
        train_df, cfg['data']['train_dir'], 
        transform=train_tf, normalize=normalize,
        do_mixup=cfg['data']['use_fourier_mixup'],
        mixup_settings=mixup_cfg,
        img_size=cfg['data']['img_size']
    )
    
    val_ds = CRCTileDataset(
        val_df, cfg['data']['train_dir'],
        transform=val_tf, normalize=normalize,
        do_mixup=False,
        img_size=cfg['data']['img_size']
    )
    
    # Sampler
    y_train = train_df['label'].values
    sample_weights = get_class_weights(y_train)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg['training']['batch_size'],
        sampler=sampler, num_workers=cfg['system']['num_workers'],
        pin_memory=cfg['system']['pin_memory']
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['training']['batch_size'],
        shuffle=False, num_workers=cfg['system']['num_workers'],
        pin_memory=cfg['system']['pin_memory']
    )
    
    # 3. Model
    print("Building SpectraViT model...")
    model = SpectraViT(
        num_classes=cfg['data']['num_classes'],
        img_size=cfg['data']['img_size'],
        backbone_name=cfg['model']['backbone'],
        unfreeze_blocks=cfg['model']['unfreeze_pattern']
    ).to(device)
    
    # 4. Optimizer (Split Parameter Groups)
    # Group 1: Classifier / Heads / ASG / Norms (High LR)
    # Group 2: Backbone (Low LR)
    backbone_ids = list(map(id, model.backbone.parameters()))
    backbone_params = filter(lambda p: id(p) in backbone_ids and p.requires_grad, model.parameters())
    rest_params = filter(lambda p: id(p) not in backbone_ids and p.requires_grad, model.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': float(cfg['training']['optimizer']['lr_backbone'])},
        {'params': rest_params, 'lr': float(cfg['training']['optimizer']['lr_classifier'])}
    ], weight_decay=float(cfg['training']['optimizer']['weight_decay']))
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg['training']['epochs'], 
        eta_min=float(cfg['training']['scheduler']['min_lr'])
    )
    
    # Loss
    criterion = SpectraLoss(
        lambda_focal=cfg['loss']['alpha'],
        lambda_supcon=(1.0 - cfg['loss']['alpha']),
        lambda_l1=float(cfg['loss']['lambda_l1']),
        lambda_tv=float(cfg['loss']['lambda_tv']),
        temperature=float(cfg['loss']['temp'])
    ).to(device)
    
    # AMP Scaler
    scaler = torch.cuda.amp.GradScaler() if cfg['training']['mixed_precision'] else None
    
    # Logging
    log_dir = os.path.join("outputs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Save config
    with open(os.path.join(log_dir, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f)
        
    print(f"Starting training for {cfg['training']['epochs']} epochs...")
    
    best_f1 = 0.0
    
    # Dry Run logic
    if args.dry_run:
        print("[Dry Run] Running 1 batch training...")
        # Break dataset to 1 item or just break loop
        # We'll just run one step of train and val
        train_one_epoch(model, [next(iter(train_loader))], criterion, optimizer, device, current_aux_weight=0.5, scaler=scaler)
        print("[Dry Run] Complete.")
        return

    # Loop
    for epoch in range(1, cfg['training']['epochs'] + 1):
        # Aux Schedule
        target_aux = cfg['loss']['aux_target']
        # Linear warm -> Hold -> Linear decay
        if epoch <= 5:
            curr_aux = target_aux * (epoch / 5.0)
        elif epoch < 0.75 * cfg['training']['epochs']:
            curr_aux = target_aux
        else:
            decay_steps = cfg['training']['epochs'] - int(0.75 * cfg['training']['epochs'])
            progress = (epoch - int(0.75 * cfg['training']['epochs'])) / decay_steps
            curr_aux = target_aux - (target_aux - 0.1) * progress
        
        train_stats, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            current_aux_weight=curr_aux, scaler=scaler
        )
        
        val_loss, val_f1 = validate(model, val_loader, criterion, device)
        
        # Step Scheduler
        scheduler.step()
        
        # Log
        print(f"Epoch {epoch} | Train Loss: {train_stats['loss_total']:.4f} F1: {train_f1:.4f} | Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")
        
        writer.add_scalar("F1/Train", train_f1, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)
        for k,v in train_stats.items():
            writer.add_scalar(f"Loss/{k}", v, epoch)
            
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print("  [+] Saved Best Model")
            
    print(f"Done. Best Val F1: {best_f1}")
    torch.save(model.state_dict(), os.path.join(log_dir, "last_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config yaml")
    parser.add_argument("--exp_name", default="run_001", help="Experiment name")
    parser.add_argument("--dry_run", action="store_true", help="Run a single batch smoke test")
    args = parser.parse_args()
    main(args)
