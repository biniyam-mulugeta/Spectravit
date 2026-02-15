import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Add project root to path to import src
# Script is in ablation_backbone/dinov2/visualizations/
# Root is ../../../
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.spectravit import SpectraViT

# --- Helper Functions ---
def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should be np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class SpectralGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple, usually (grad,)
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        self.model.zero_grad()
        logits, _, _, _ = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        probs = F.softmax(logits, dim=1)
        confidence = probs[0, class_idx].item()
            
        # 2. Backward Pass
        score = logits[0, class_idx]
        score.backward()
        
        # 3. Generate CAM
        # Gradients: [B, C, H, W]
        # Activations: [B, C, H, W]
        
        # Global Average Pooling of Gradients -> Weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # [B, C, 1, 1]
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [B, 1, H, W]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)
        
        return cam.cpu().numpy(), class_idx, confidence

class AttentionRollout:
    def __init__(self, model):
        self.model = model
        self.attention_map = None
        
        # Disable fused attention in the last block to ensure the hook fires.
        # F.scaled_dot_product_attention bypasses the attn_drop layer.
        last_block = model.backbone.blocks[-1]
        if hasattr(last_block.attn, 'fused_attn'):
            last_block.attn.fused_attn = False

        # Hook into the dropout layer of the attention block in the last layer
        # timm structure: model.backbone.blocks[-1].attn.attn_drop
        # The input to attn_drop is the softmaxed attention matrix
        self.hook = last_block.attn.attn_drop.register_forward_hook(self.save_attention)

    def save_attention(self, module, input, output):
        # input[0] is the attention matrix of shape [B, NumHeads, N, N]
        self.attention_map = input[0].detach()

    def get_attention(self):
        if self.attention_map is None:
            raise RuntimeError("Attention map not captured. Fused attention might still be active.")
            
        # Average over heads
        # Shape: [B, N, N]
        attn = torch.mean(self.attention_map, dim=1)
        
        # Get CLS token attention to all other tokens
        # CLS is at index 0
        # attn[0, 0, 1:] -> Attention of CLS to patches
        cls_attn = attn[:, 0, 1:] # [B, NumPatches]
        
        # Reshape to grid
        # Assuming square grid
        num_patches = cls_attn.shape[1]
        grid_size = int(np.sqrt(num_patches))
        
        cls_attn = cls_attn.reshape(-1, grid_size, grid_size)
        
        # Normalize
        cls_attn = cls_attn - torch.min(cls_attn)
        cls_attn = cls_attn / (torch.max(cls_attn) + 1e-7)
        
        return cls_attn.cpu().numpy()

def process_image(model, grad_cam, attn_extractor, image_path, output_dir, device):
    # 3. Load Image
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = img_pil.resize((224, 224))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_pil).unsqueeze(0).to(device).requires_grad_(True)
    
    # 4. Run Inference & Visualization
    print(f"Processing {os.path.basename(image_path)}...")

    # Grad-CAM (requires backward)
    cam_map, pred_idx, confidence = grad_cam(input_tensor) # [1, 1, H, W] (16x16)
    
    # Attention (captured during forward pass of Grad-CAM)
    attn_map = attn_extractor.get_attention() # [1, 16, 16]
    
    # 5. Process Maps for Plotting
    cam_map = cam_map[0, 0] # 16x16
    attn_map = attn_map[0]  # 16x16
    
    # Upsample to 224x224
    cam_resized = cv2.resize(cam_map, (224, 224))
    attn_resized = cv2.resize(attn_map, (224, 224))
    
    # Prepare Original Image
    img_np = np.array(img_pil) / 255.0
    
    # Create Overlays
    cam_overlay = show_cam_on_image(img_np, cam_resized, use_rgb=True)
    attn_overlay = show_cam_on_image(img_np, attn_resized, use_rgb=True, colormap=cv2.COLORMAP_VIRIDIS)
    
    # Combined Overlay (Average of both masks)
    combined_mask = (cam_resized + attn_resized) / 2.0
    combined_mask = combined_mask / np.max(combined_mask)
    combined_overlay = show_cam_on_image(img_np, combined_mask, use_rgb=True)

    # 6. Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(img_pil)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    axs[1].imshow(cam_overlay)
    axs[1].set_title(f"Spectral Grad-CAM)")
    axs[1].axis('off')
    
    axs[2].imshow(attn_overlay)
    axs[2].set_title("DINOv2 Self-Attention")
    axs[2].axis('off')
    
    axs[3].imshow(combined_overlay)
    axs[3].set_title("Combined Overlay")
    axs[3].axis('off')
    
    fig.suptitle(f"Prediction: Confidence: {confidence:.2%}", fontsize=16)
    
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{filename}_spectral_vis.png")
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Loading SpectraViT from {args.ckpt}...")
    model = SpectraViT(num_classes=args.num_classes, img_size=224)
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    # 2. Setup Hooks
    # Target the Spectral Gating module for Grad-CAM
    grad_cam = SpectralGradCAM(model, model.spectral_gating)
    
    # Target the Backbone Attention for Self-Attention
    attn_extractor = AttentionRollout(model)
    
    # 3. Determine Images
    image_paths = []
    if os.path.isdir(args.image_path):
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        class_map = {}
        
        for root, _, files in os.walk(args.image_path):
            for f in files:
                if f.lower().endswith(valid_exts) and not f.startswith('.'):
                    full_path = os.path.join(root, f)
                    class_name = os.path.basename(os.path.dirname(full_path))
                    class_map.setdefault(class_name, []).append(full_path)
        
        for class_name, paths in sorted(class_map.items()):
            image_paths.extend(paths[:args.images_per_class])
            
    elif os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        print(f"Error: {args.image_path} is not a valid file or directory.")
        return

    print(f"Processing {len(image_paths)} images...")
    for img_p in image_paths:
        process_image(model, grad_cam, attn_extractor, img_p, args.output_dir, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True, help="Path to input image or directory")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="visualizations/outputs", help="Output directory")
    parser.add_argument("--num_classes", type=int, default=9)
    parser.add_argument("--images_per_class", type=int, default=5, help="Number of images per class to generate")
    args = parser.parse_args()
    main(args)