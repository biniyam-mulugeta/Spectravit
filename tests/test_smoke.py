import unittest
import torch
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.spectravit import SpectraViT
from src.losses.spectra_loss import SpectraLoss
from src.data.transforms import get_transforms

class TestSpectraViT(unittest.TestCase):
    def test_model_instantiation(self):
        """Test if model builds without error and produces correct output shape."""
        model = SpectraViT(num_classes=9, img_size=224, backbone_name='vit_small_patch14_dinov2.lvd142m')
        self.assertIsNotNone(model)
        
        # Test Forward
        dummy_in = torch.randn(2, 3, 224, 224)
        logits, emb, spec, aux = model(dummy_in)
        
        self.assertEqual(logits.shape, (2, 9))
        self.assertEqual(aux.shape, (2, 9))
        
    def test_loss_forward(self):
        """Test loss computation."""
        criterion = SpectraLoss(lambda_focal=0.5, lambda_l1=0.1)
        
        logits = torch.randn(2, 9)
        emb = torch.randn(2, 128)
        spec = torch.randn(2, 384, 16, 16)
        aux = torch.randn(2, 9)
        labels = torch.tensor([0, 1])
        
        loss, loss_dict = criterion(logits, emb, spec, aux, labels, current_aux_weight=0.5)
        self.assertTrue(loss > 0)
        self.assertIn("loss_focal", loss_dict)

if __name__ == '__main__':
    unittest.main()
