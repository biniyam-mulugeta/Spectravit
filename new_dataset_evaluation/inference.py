import os
import sys
import argparse
import yaml
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.spectravit import SpectraViT
from src.data.transforms import get_transforms

def main():
    parser = argparse.ArgumentParser(description="Run inference on new dataset")
    parser.add_argument("--data_dir", required=True, help="Path to folder of images")
    parser.add_argument("--ckpt", required=True, help="Path to trained model.pth")
    parser.add_argument("--class_map", default=None, help="Path to class_map.yaml (optional, else uses sorted folders/digits)")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Determine Classes
    # Try to load adjacent class map if not provided
    if args.class_map is None:
        ckpt_dir = os.path.dirname(args.ckpt)
        cand = os.path.join(ckpt_dir, "class_map.yaml")
        if os.path.exists(cand):
            args.class_map = cand
            
    if args.class_map:
        with open(args.class_map, 'r') as f:
            class_map = yaml.safe_load(f)
            # Invert: id -> name
            idx_to_class = {v: k for k, v in class_map.items()}
            num_classes = len(class_map)
            print(f"Loaded class map: {num_classes} classes")
    else:
        # Fallback: Inference without names (just indices)
        print("Warning: No class_map found. Predicting indices only.")
        num_classes = 9 # Default? Or try to guess? 
        # Better to error out or require explicit num_classes arg?
        # Let's peek at state dict
        state = torch.load(args.ckpt, map_location='cpu')
        w = state['classifier.weight']
        num_classes = w.shape[0]
        idx_to_class = {i: str(i) for i in range(num_classes)}
        print(f"Inferred {num_classes} classes from checkpoint.")

    # 2. Model
    model = SpectraViT(num_classes=num_classes, img_size=args.img_size)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.to(device)
    model.eval()

    # 3. Data Scan
    # Flatten folder
    images = []
    print(f"Scanning {args.data_dir}...")
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                images.append(os.path.join(root, f))
                
    if not images:
        print("No images found.")
        return
        
    print(f"Found {len(images)} images.")
    
    # 4. Loop
    _, val_tf, normalize = get_transforms(args.img_size)
    # Just need resize/norm
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    results = []
    batch_size = 32
    
    # Simple batching
    for i in tqdm(range(0, len(images), batch_size)):
        batch_paths = images[i:i+batch_size]
        batch_tensors = []
        valid_batch_paths = []
        
        for p in batch_paths:
            try:
                img = Image.open(p).convert('RGB')
                batch_tensors.append(tf(img))
                valid_batch_paths.append(p)
            except Exception as e:
                print(f"Error reading {p}: {e}")
                
        if not batch_tensors: continue
        
        batch_stack = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            logits = model(batch_stack)
            if isinstance(logits, tuple): logits = logits[0]
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
        for path, pred_idx in zip(valid_batch_paths, preds):
            results.append({
                "file": path,
                "prediction_index": pred_idx,
                "prediction_label": idx_to_class.get(pred_idx, str(pred_idx))
            })
            
    # Save
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")

if __name__ == "__main__":
    main()
