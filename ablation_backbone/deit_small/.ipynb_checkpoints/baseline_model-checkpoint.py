import torch
import torch.nn as nn
import timm

class BaselineModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.backbone_name = "deit_small_patch16_224"
        self.backbone = timm.create_model(self.backbone_name, pretrained=True, num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)
