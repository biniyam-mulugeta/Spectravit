import argparse
import torch
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

# Imports from local folder
from baseline_model import BaselineModel
from spectragate_model import SpectraGateModel

# Add project root to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.spectravit import SpectraViT

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading {args.model}...")
    if args.model == 'baseline':
        model = BaselineModel(num_classes=9)
    elif args.model == 'spectragate':
        model = SpectraGateModel(num_classes=9)
    elif args.model == 'spectravit':
        model = SpectraViT(num_classes=9)
        
    # Load Weights
    # Strict=False to allow for minor mismatch if user provides partial weights,
    # but strictly user said "ckpt.pth".
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        print("Weights loaded.")
    else:
        print(f"Checkpoint {args.ckpt} not found. Running with random weights (Sanity Mode).")

    model.to(device)
    model.eval()

    # Data
    valid_exts = ('.tif', '.png', '.jpg')
    paths = []
    for root, dirs, files in os.walk(args.data):
        # Skip hidden directories like .ipynb_checkpoints
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in files:
            if f.lower().endswith(valid_exts):
                paths.append(os.path.join(root, f))
                
    if not paths and os.path.isfile(args.data):
        paths = [args.data]
        
    print(f"Found {len(paths)} images.")

    # Transform (Standard 224)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Helper to get ground truth from path, assuming `.../class_name/image.ext`
    def get_true_label(path):
        return os.path.basename(os.path.dirname(path))

    # Data containers
    file_paths = []
    true_labels = []
    predictions = []
    probabilities = []
    embeddings = []

    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert('RGB')
            im_t = tf(img).unsqueeze(0).to(device)
            
            out = model(im_t)
            
            logits = out
            embedding = None

            # Handle tuple return from SpectraGate
            if args.model == 'spectragate':
                # Assumes model returns (logits, embedding, aux_logits)
                logits, embedding, _ = out
            elif args.model == 'spectravit':
                # SpectraViT returns (logits, embeddings, spectral_feat, aux_logits)
                logits, embedding, _, _ = out
            elif isinstance(out, tuple): # Fallback for other models
                logits = out[0] 
                
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
            # Store results
            file_paths.append(p)
            true_labels.append(get_true_label(p))
            predictions.append(pred)
            probabilities.append(probs.cpu().numpy().flatten())
            if embedding is not None:
                embeddings.append(embedding.cpu().numpy().flatten())
            
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, f"{args.model}_inference_results.npz")
    
    np.savez_compressed(out_path, 
                        files=file_paths, true_labels=true_labels, predictions=predictions, 
                        probabilities=np.array(probabilities), embeddings=np.array(embeddings))
    
    print(f"Saved inference results to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['baseline', 'spectragate', 'spectravit'], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/")
    args = parser.parse_args()
    main(args)
