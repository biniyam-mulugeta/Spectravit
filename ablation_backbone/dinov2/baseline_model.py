import torch
import torch.nn as nn
import timm

class BaselineModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone_name = "vit_small_patch14_dinov2.lvd142m"
        # Standard Linear Probing / Finetuning Interface
        # Num classes=0 removes head
        self.backbone = timm.create_model(self.backbone_name, pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)
