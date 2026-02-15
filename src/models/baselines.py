import torch
import torch.nn as nn
import timm

class BaselineModel(nn.Module):
    """
    Standard Transfer Learning Wrapper.
    """
    def __init__(self, backbone_name: str, num_classes: int = 9, img_size: int = 224, pretrained: bool = True):
        super().__init__()
        # Load model with correct num_classes immediately if possible, 
        # or remove head and add own linear layer.
        # Simple approach: let timm handle head if allowed, or replace it.
        
        try:
            self.model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes, img_size=img_size)
        except TypeError:
            # Some models don't take img_size
            self.model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
            
    def forward(self, x):
        return self.model(x)
