# New Dataset Evaluation

This module provides tools to evaluate SpectraViT on any new dataset (e.g., BACH).

## Structure
- `train_scratch.py`: Train end-to-end from scratch.
- `train_transfer.py`: Fine-tune a pretrained model (Transfer Learning).
- `inference.py`: Run prediction on a folder of images.

## 1. Preparing Data
Organize your dataset in the standard ImageFolder format:
```
dataset_root/
    class_A/
        img1.jpg
        img2.jpg
    class_B/
        img3.jpg
        ...
```

## 2. Train from Scratch
Use this if you have a large dataset and want to train a fresh model.
```bash
python new_dataset_evaluation/train_scratch.py \
  --data_dir /path/to/dataset \
  --output_dir outputs/my_new_model \
  --epochs 30
```

## 3. Transfer Learning (Fine-Tuning)
Use this if you have a smaller dataset (like BACH) and want to adapt a pretrained model.
```bash
python new_dataset_evaluation/train_transfer.py \
  --data_dir /path/to/dataset \
  --ckpt outputs/spectravit_run_01/best_model.pth \
  --output_dir outputs/my_transfer_model \
  --freeze_backbone \
  --epochs 20
```
*   `--freeze_backbone`: Keeps the transformer weights fixed, training only the head. Remove this flag to unfreeze everything.

## 4. Inference
Run prediction on a folder of images (nested or flat).
```bash
python new_dataset_evaluation/inference.py \
  --data_dir /path/to/test_images \
  --ckpt outputs/my_transfer_model/best_model.pth \
  --output my_predictions.csv
```
It will automatically look for `class_map.yaml` in the checkpoint folder to map indices back to class names.
