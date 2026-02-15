import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.utils.seeding import seed_everything
from src.data.dataset import CRCTileDataset, scan_folder_to_df
from src.data.transforms import get_transforms
from src.train.trainer import train_one_epoch, validate
# Use shared SpectraLoss for consistency (Constraint: "DO NOT change SpectraLoss")
from src.losses.spectra_loss import SpectraLoss

def get_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    weight_per_class = 1.0 / counts
    weights_lookup = dict(zip(classes, weight_per_class))
    return np.array([weights_lookup[cls] for cls in y])

def get_ablation_model(variant: str, model_type: str, num_classes: int):
    """
    Dynamically imports the requested model from the specific folder.
    variant: dinov2, deit_small, vit_small, mae_vit_small
    model_type: baseline, spectragate
    """
    root = os.path.dirname(__file__)
    variant_path = os.path.join(root, variant)
    if not os.path.exists(variant_path):
        raise ValueError(f"Variant folder {variant} not found in {root}")
    
    # Import machinery
    sys.path.insert(0, variant_path)
    try:
        if model_type == 'baseline':
            from baseline_model import BaselineModel
            model = BaselineModel(num_classes=num_classes)
        elif model_type == 'spectragate':
            from spectragate_model import SpectraGateModel
            model = SpectraGateModel(num_classes=num_classes)
        else:
            raise ValueError("model_type must be baseline or spectragate")
    finally:
        sys.path.pop(0)
        # Clean sys.modules to prevent collisions between variants
        for k in ['baseline_model', 'spectragate_model']:
            if k in sys.modules: del sys.modules[k]
            
    return model

def main(args):
    # 1. Config
    # Locate config relative to script if default fails
    config_path = args.config
    if not os.path.exists(config_path):
        # Try resolving relative to project root (2 levels up from this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir) # ablation_backbone -> root
        alt_path = os.path.join(project_root, args.config)
        if os.path.exists(alt_path):
            config_path = alt_path
            
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    seed_everything(cfg['system']['seed'])
    device = torch.device(cfg['system']['device'])
    
    experiment_name = f"{args.variant}_{args.model}_run"
    print(f"=== Starting Ablation Training: {args.variant} / {args.model} ===")
    
    # 2. Model
    model = get_ablation_model(args.variant, args.model, cfg['data']['num_classes'])
    model = model.to(device)
    
    # 3. Data (Reused from Main Pipeline)
    print(f"Loading data from {cfg['data']['train_dir']}")
    df = scan_folder_to_df(cfg['data']['train_dir'], cfg['data']['class_map'])
    
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=cfg['system']['seed'], stratify=df['label']
    )
    
    train_tf, val_tf, normalize = get_transforms(cfg['data']['img_size'])
    # Constraint: "hyperparameters" unchanged. Use config.
    mixup_cfg = {
        "prob": cfg['data']['mixup_prob'],
        "beta": cfg['data']['mixup_beta'],
        "strategy": cfg['data']['mixup_strategy']
    }
    
    train_ds = CRCTileDataset(train_df, cfg['data']['train_dir'], transform=train_tf, normalize=normalize, 
                              do_mixup=cfg['data']['use_fourier_mixup'], mixup_settings=mixup_cfg, img_size=cfg['data']['img_size'])
    val_ds = CRCTileDataset(val_df, cfg['data']['train_dir'], transform=val_tf, normalize=normalize, do_mixup=False, img_size=cfg['data']['img_size'])
    
    sampler = WeightedRandomSampler(get_class_weights(train_df['label'].values), len(train_df), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)
    
    # 4. Optimizer (Constraint: Same Split LR)
    # Both Baseline and SpectraGate wrapper have .backbone
    if hasattr(model, 'backbone'):
        backbone_ids = list(map(id, model.backbone.parameters()))
        backbone_params = filter(lambda p: id(p) in backbone_ids and p.requires_grad, model.parameters())
        rest_params = filter(lambda p: id(p) not in backbone_ids and p.requires_grad, model.parameters())
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': float(cfg['training']['optimizer']['lr_backbone'])},
            {'params': rest_params, 'lr': float(cfg['training']['optimizer']['lr_classifier'])}
        ], weight_decay=float(cfg['training']['optimizer']['weight_decay']))
    else:
        # Fallback if backbone not identified (should not happen based on our impl)
        print("Warning: Could not identify .backbone for split LR. Using default LR.")
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'], eta_min=float(cfg['training']['scheduler']['min_lr']))
    
    # 5. Loss (Constraint: Same SpectraLoss)
    criterion = SpectraLoss(
        lambda_focal=cfg['loss']['alpha'],
        lambda_supcon=(1.0 - cfg['loss']['alpha']),
        lambda_l1=float(cfg['loss']['lambda_l1']),
        lambda_tv=float(cfg['loss']['lambda_tv']),
        temperature=float(cfg['loss']['temp'])
    ).to(device)
    
    scaler = torch.cuda.amp.GradScaler() if cfg['training']['mixed_precision'] else None
    
    # 6. Loop
    output_dir = os.path.join("outputs", "ablation", experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    best_f1 = 0.0
    for epoch in range(1, cfg['training']['epochs'] + 1):
        # Aux Schedule reused
        target_aux = cfg['loss']['aux_target']
        if epoch <= 5: curr_aux = target_aux * (epoch / 5.0)
        elif epoch < 0.75 * cfg['training']['epochs']: curr_aux = target_aux
        else:
            decay_steps = cfg['training']['epochs'] - int(0.75 * cfg['training']['epochs'])
            progress = (epoch - int(0.75 * cfg['training']['epochs'])) / decay_steps
            curr_aux = target_aux - (target_aux - 0.1) * progress
            
        # Baseline model returns simple logits -> trainer handles tuple check
        # But baseline model output is NOT a tuple. Trainer needs to handle this.
        # My src/train/trainer.py DOES handle `isinstance(output, tuple)`.
        # However, for Baseline, we need to ensure SpectraLoss handles simple input too.
        # SpectraLoss.forward signature: (logits_main, embeddings, spectral_feat, logits_aux, labels, ...)
        # Baseline output: Just logits.
        # We need a small Adapter here if model is baseline, or logic in loop.
        
        # Let's use a lambda wrapper for criterion if baseline.
        if args.model == 'baseline':
            # Baseline doesn't output embeddings/aux/spectral.
            # SpectraLoss expects them.
            # We must use standard FocalLoss for Baseline to be fair?
            # User constraint: "DO NOT change SpectraLoss". 
            # But SpectraLoss requires spectral features. Baseline HAS NO spectral features.
            # So Baseline must use standard Focal Loss (or implied logic).
            # We'll use the .focal component of SpectraLoss only.
            
            def baseline_criterion(logits, labels, **kwargs):
                loss = criterion.focal(logits, labels)
                return loss
            
            epoch_crit = baseline_criterion
        else:
            # SpectraGate
            epoch_crit = criterion

        stats, train_f1 = train_one_epoch(
            model, train_loader, epoch_crit, optimizer, device, 
            current_aux_weight=curr_aux, scaler=scaler
        )
        
        # Validation
        v_loss, v_f1 = validate(model, val_loader, epoch_crit, device)
        scheduler.step()
        
        print(f"Epoch {epoch} | Train F1: {train_f1:.4f} | Val F1: {v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            
    print(f"Done. Best F1: {best_f1}. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variant", choices=['dinov2', 'deit_small', 'vit_small', 'mae_vit_small'], required=True)
    parser.add_argument("--model", choices=['baseline', 'spectragate'], required=True)
    args = parser.parse_args()
    main(args)
