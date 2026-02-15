import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np

def train_one_epoch(model, loader, criterion, optimizer, device, current_aux_weight=0.0, scaler=None):
    model.train()
    
    losses_tracker = {}
    all_preds, all_labels = [], []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        
        # AMP Context
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            output = model(images)
            if isinstance(output, (tuple, list)):
                logits, embeddings, spectral_feat, aux_logits = output
                total_loss, loss_dict = criterion(
                    logits, embeddings, spectral_feat, aux_logits, labels, 
                    current_aux_weight=current_aux_weight
                )
            else:
                # Baseline
                logits = output
                total_loss = criterion(logits, labels) # Baseline uses FocalLoss direct
                loss_dict = {"loss_total": total_loss}

        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Track
        for k, v in loss_dict.items():
            losses_tracker.setdefault(k, []).append(v.item())
            
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    # Averages
    avg_losses = {k: np.mean(v) for k, v in losses_tracker.items()}
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_losses, f1

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses_tracker = {}
    all_preds, all_labels = [], []
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.amp.autocast('cuda'):
            output = model(images)
            
            if isinstance(output, (tuple, list)):
                logits, embeddings, spectral_feat, aux_logits = output
                # No aux loss in validation usually, or weight 0
                total_loss, loss_dict = criterion(
                    logits, embeddings, spectral_feat, aux_logits, labels, 
                    current_aux_weight=0.0
                )
            else:
                logits = output
                total_loss = criterion(logits, labels)
                loss_dict = {"loss_total": total_loss}
        
        losses_tracker.setdefault("loss_total", []).append(total_loss.item())
        
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        
    avg_loss = np.mean(losses_tracker["loss_total"])
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1
