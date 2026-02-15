import os
import sys
import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.spectravit import SpectraViT
from src.data.transforms import get_transforms

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        
        # Scan folder (flat or nested, doesn't matter, we just find images)
        valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_exts):
                    self.paths.append(os.path.join(root, file))
                    
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

def predict(args):
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device(cfg['system']['device'])
    
    # Load Model
    print(f"Loading model from {args.ckpt}")
    model = SpectraViT(
        num_classes=cfg['data']['num_classes'],
        img_size=cfg['data']['img_size'],
        backbone_name=cfg['model']['backbone']
    ).to(device)
    
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    
    # Dataset
    _, val_tf, normalize = get_transforms(cfg['data']['img_size'])
    # Combine (since get_transforms returns separate norm)
    # Actually get_transforms returns (pre_train, pre_val, norm)
    # We need to construct full val transform
    full_transform = transforms.Compose([
        val_tf,
        normalize
    ])
    
    ds = InferenceDataset(args.data_dir, transform=full_transform)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)
    
    print(f"Found {len(ds)} images in {args.data_dir}")
    
    results = []
    idx_to_class = {v: k for k, v in cfg['data']['class_map'].items()}
    
    print("Running prediction...")
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            output = model(images)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            
            for i in range(len(paths)):
                pred_label = idx_to_class[preds_np[i]]
                confidence = probs_np[i][preds_np[i]]
                
                res = {
                    "file_path": paths[i],
                    "prediction": pred_label,
                    "confidence": float(confidence)
                }
                # Add all probabilities
                for cls_idx, cls_name in idx_to_class.items():
                    res[f"prob_{cls_name}"] = float(probs_np[i][cls_idx])
                    
                results.append(res)
                
    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "predictions.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True, help="Folder of images to predict")
    parser.add_argument("--out_dir", default="inference_outputs")
    args = parser.parse_args()
    predict(args)
