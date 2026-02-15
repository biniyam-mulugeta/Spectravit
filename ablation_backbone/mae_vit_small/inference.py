import argparse
import torch
import os
import sys
import pandas as pd
from PIL import Image
from torchvision import transforms

from baseline_model import BaselineModel
from spectragate_model import SpectraGateModel

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {args.model}...")
    if args.model == 'baseline': model = BaselineModel(num_classes=9)
    else: model = SpectraGateModel(num_classes=9)
    
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Weights loaded.")
    else: print(f"Checkpoint {args.ckpt} not found. Running random weights.")

    model.to(device)
    model.eval()

    valid_exts = ('.tif', '.png', '.jpg')
    paths = []
    for root, _, files in os.walk(args.data):
        for f in files:
            if f.lower().endswith(valid_exts): paths.append(os.path.join(root, f))
    if not paths and os.path.isfile(args.data): paths = [args.data]
    print(f"Found {len(paths)} images.")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    results = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert('RGB')
            im_t = tf(img).unsqueeze(0).to(device)
            out = model(im_t)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            results.append({"file": p, "prediction": pred, "confidence": probs[0, pred].item()})
            
    os.makedirs(args.output, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(args.output, f"{args.model}_predictions.csv"), index=False)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['baseline', 'spectragate'], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    main(args)
