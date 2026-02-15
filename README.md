# SpectraViT Reproduction Project

This repository contains a clean, reproducible implementation of **SpectraViT** (DINOv2 + Adaptive Spectral Gating) for Histopathology Image Analysis.

## Directory Structure
- `src/`: Core source code (models, data, losses, training engine).
- `scripts/`: Executable scripts for training and evaluation.
- `configs/`: Hyperparameter configurations.
- `outputs/`: Training logs and checkpoints.

## 1. Getting Started

### Prerequisites
- Python 3.8+
- Conda (optional, but recommended)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/biniyam-mulugeta/Spectravit.git
    cd Spectravit
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Tests
To verify the installation and that everything is working as expected, run the smoke tests:
```bash
python scripts/train_crc.py --dry_run
```
Or run the unit tests:
```bash
python -m unittest discover tests
```

## 2. Usage Guidelines

### A. Train on CRC-100K
```bash
python scripts/train_crc.py --config configs/default.yaml --exp_name spectravit_run_01
```
Outputs (checkpoints, logs) will be in `outputs/spectravit_run_01/`.

### B. Evaluate on CRC-7K
```bash
python scripts/eval_crc7k.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/CRC-VAL-HE-7K
```

### C. Robustness Evaluation
Evaluates model against 5 levels of corruption (Blur, Noise, Color Jitter).
```bash
python scripts/robustness_eval.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/CRC-VAL-HE-7K
```

### D. Transfer Learning (BACH)
**Linear Probe (Frozen Backbone):**
```bash
python scripts/transfer_bach_linear.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/BACH
```

**Full Finetune:**
```bash
python scripts/transfer_bach_finetune.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/BACH
```


### E. Generic Inference (Unlabeled Data)
Run predictions on any folder of images (nested or flat).
```bash
python scripts/predict.py --config configs/default.yaml --ckpt outputs/spectravit_run_01/best_model.pth --data_dir /path/to/any/images
```
Outputs a CSV with class probabilities to `inference_outputs/predictions.csv`.

## 3. Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to report a bug or suggest a new feature.

### Pull Requests
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a pull request.

### Reporting Bugs
Please open an issue with a clear description of the bug, including steps to reproduce it.

## 4. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
