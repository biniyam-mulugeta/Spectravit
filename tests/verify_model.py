import sys
import os
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.spectravit import SpectraViT

def verify():
    print("Initializing SpectraViT (9 classes)...")
    try:
        model = SpectraViT(num_classes=9)
    except Exception as e:
        print(f"FAIL: Initialization error: {e}")
        return

    print("Checking attributes...")
    has_main = hasattr(model, 'classifier')
    has_aux = hasattr(model, 'aux_spec_classifier')
    has_wrong = hasattr(model, 'aux_classifier')
    
    print(f"Has 'classifier': {has_main}")
    print(f"Has 'aux_spec_classifier': {has_aux}")
    print(f"Has 'aux_classifier' (should be False): {has_wrong}")
    
    if not has_main:
        print("FAIL: Missing 'classifier' attribute.")
    if not has_aux:
        print("FAIL: Missing 'aux_spec_classifier' attribute.")
        
    # Check forward pass
    print("Running forward pass (B=2, 3, 224, 224)...")
    try:
        x = torch.randn(2, 3, 224, 224)
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            
        logits, emb, spec, aux = model(x)
        
        print(f"Logits shape: {logits.shape}")
        print(f"Aux logits shape: {aux.shape}")
        
        if logits.shape[1] == 9 and aux.shape[1] == 9:
            print("PASS: Shapes correct (9 classes).")
        else:
            print(f"FAIL: Shape mismatch. Expected 9, got {logits.shape[1]} and {aux.shape[1]}")
            
    except Exception as e:
        print(f"FAIL: Forward pass error: {e}")

if __name__ == "__main__":
    verify()
