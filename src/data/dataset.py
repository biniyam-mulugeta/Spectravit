import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def fourier_mixup(img_tensor_1: torch.Tensor, img_tensor_2: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Blend amplitude spectra while keeping phase from img 1 (structure-preserving).
    img_tensor_1, img_tensor_2: [3, H, W]
    """
    lam = np.random.beta(beta, beta)
    img1 = img_tensor_1.to(torch.float32)
    img2 = img_tensor_2.to(torch.float32)

    # FFT per channel
    fft1 = torch.fft.fft2(img1, dim=(-2, -1))
    fft2 = torch.fft.fft2(img2, dim=(-2, -1))

    amp1, phase1 = torch.abs(fft1), torch.angle(fft1)
    amp2 = torch.abs(fft2)

    mixed_amp = lam * amp1 + (1 - lam) * amp2
    mixed_fft = mixed_amp * torch.exp(1j * phase1)
    mixed = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real

    return mixed.clamp(0, 1)

class CRCTileDataset(Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 root_dir: str, 
                 transform=None, 
                 normalize=None, 
                 do_mixup: bool = False, 
                 mixup_settings: dict = None,
                 max_read_tries: int = 10,
                 img_size: int = 224):
        
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.paths = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()
        self.transform = transform
        self.normalize = normalize
        self.do_mixup = do_mixup
        self.max_read_tries = max_read_tries
        self.img_size = img_size
        
        # Mixup Config
        self.mixup_prob = 0.0
        self.mixup_beta = 1.0
        self.mixup_strategy = "random"
        
        if mixup_settings:
            self.mixup_prob = mixup_settings.get("prob", 0.5)
            self.mixup_beta = mixup_settings.get("beta", 1.0)
            self.mixup_strategy = mixup_settings.get("strategy", "cross_class")

    def __len__(self):
        return len(self.paths)

    def _read_rgb(self, path: str):
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _safe_get_image(self, idx: int):
        for _ in range(self.max_read_tries):
            path = os.path.join(self.root_dir, self.paths[idx])
            img = self._read_rgb(path)
            if img is not None:
                return img, self.labels[idx]
            # Fallback to random
            idx = np.random.randint(0, len(self.paths))
            
        # Hard fallback: black image
        return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8), self.labels[idx]

    def _sample_partner_index(self, label: int):
        if self.mixup_strategy == "same_class":
            candidates = np.where(np.array(self.labels) == label)[0]
            if len(candidates) > 0: return int(np.random.choice(candidates))
        elif self.mixup_strategy == "cross_class":
            candidates = np.where(np.array(self.labels) != label)[0]
            if len(candidates) > 0: return int(np.random.choice(candidates))
            
        # Default random
        return int(np.random.randint(0, len(self.paths)))

    def __getitem__(self, idx):
        img, label = self._safe_get_image(idx)

        # 1. Transform (Resize, Flip, etc.) -> Tensor [0,1]
        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        # 2. Fourier Mixup (Phase-Preserving)
        if self.do_mixup and np.random.random() < self.mixup_prob:
            partner_idx = self._sample_partner_index(label)
            img2, _ = self._safe_get_image(partner_idx)
            img_t2 = self.transform(img2) if self.transform else transforms.ToTensor()(img2)
            
            img_t = fourier_mixup(img_t, img_t2, beta=self.mixup_beta)

        # 3. Normalize (Mean/Std)
        if self.normalize:
            img_t = self.normalize(img_t)

        return img_t, torch.tensor(label, dtype=torch.long)

def scan_folder_to_df(root_dir: str, class_map: dict) -> pd.DataFrame:
    rows = []
    print(f"[Info] Scanning {root_dir}...")
    for class_name, label in class_map.items():
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        for fn in os.listdir(class_path):
            if fn.lower().endswith(valid_exts):
                rows.append({
                    "path": os.path.join(class_name, fn), 
                    "label": label,
                    "class_name": class_name
                })
    print(f"[Info] Found {len(rows)} images.")
    return pd.DataFrame(rows)
