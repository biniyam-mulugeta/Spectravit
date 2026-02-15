import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        else: return focal_loss.sum()

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(mask.shape[0]).view(-1, 1).to(mask.device), 0)
        mask = mask * logits_mask
        exp_sim = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        mask_sum = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-8)
        return -mean_log_prob_pos.mean()

class SpectraLoss(nn.Module):
    def __init__(self, lambda_focal=0.5, lambda_l1=1e-4, lambda_tv=5e-5, temperature=0.07):
        super().__init__()
        self.focal = FocalLoss(gamma=2.0)
        self.supcon = SupervisedContrastiveLoss(temperature)
        self.lambda_main = lambda_focal
        self.lambda_l1 = lambda_l1
        self.lambda_tv = lambda_tv

    def _total_variation(self, x):
        tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return tv_h + tv_w

    def forward(self, logits_main, embeddings, spectral_feat, logits_aux, labels, current_aux_weight=0.0):
        # 1. Main Focal
        loss_focal = self.focal(logits_main, labels)
        # 2. SupCon
        loss_supcon = self.supcon(embeddings, labels)
        # 3. Aux
        loss_aux = self.focal(logits_aux, labels) if current_aux_weight > 0 else 0.0
        # 4. Reg
        loss_reg = 0.0
        if self.lambda_l1 > 0: loss_reg += torch.mean(torch.abs(spectral_feat)) * self.lambda_l1
        if self.lambda_tv > 0: loss_reg += self._total_variation(spectral_feat) * self.lambda_tv
        
        total = self.lambda_main * loss_focal + (1-self.lambda_main) * loss_supcon + current_aux_weight * loss_aux + loss_reg
        return total, {"focal": loss_focal, "supcon": loss_supcon, "aux": loss_aux, "reg": loss_reg}
