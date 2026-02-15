import torch
import sys
import os

# Add paths
root = os.path.dirname(__file__)

backbones = ['dinov2', 'deit_small', 'vit_small', 'mae_vit_small']

def test_backbone(name):
    print(f"\n=== Testing {name} ===")
    path = os.path.join(root, name)
    sys.path.insert(0, path)
    
    try:
        from baseline_model import BaselineModel
        from spectragate_model import SpectraGateModel
    except ImportError as e:
        print(f"Failed to import {name}: {e}")
        sys.path.pop(0)
        return

    # 1. Baseline
    model_b = BaselineModel(num_classes=9)
    x = torch.randn(2, 3, 224, 224)
    out_b = model_b(x)
    print(f"Baseline Output: {out_b.shape}")
    assert out_b.shape == (2, 9)

    # 2. SpectraGate
    model_s = SpectraGateModel(num_classes=9)
    out_s = model_s(x)
    # Expect tuple (logits, spec, aux)
    assert isinstance(out_s, tuple)
    logits, spec, aux = out_s
    print(f"SpectraGate Logits: {logits.shape}")
    print(f"SpectraGate SpecFeat: {spec.shape}")
    print(f"SpectraGate Aux: {aux.shape}")
    
    assert logits.shape == (2, 9)
    assert aux.shape == (2, 9)
    # Check grid: 14x14 or 16x16
    if name == 'dinov2':
        # 224/14 = 16
        assert spec.shape[2] == 16 and spec.shape[3] == 16
    else:
        # 16
        assert spec.shape[2] == 14 and spec.shape[3] == 14
        
    print(f"SUCCESS {name}")
    
    # Clean up sys.path and modules to avoid conflicts
    sys.path.pop(0)
    # Simple way to clear local imports: remove from sys.modules
    to_rem = ['baseline_model', 'spectragate_model']
    for k in to_rem:
        if k in sys.modules: del sys.modules[k]

if __name__ == "__main__":
    for b in backbones:
        test_backbone(b)
