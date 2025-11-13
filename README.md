# DiffusionSR

A **from-scratch implementation** of the [ResShift](https://arxiv.org/abs/2307.12348) paper: an efficient diffusion-based super-resolution model that uses a U-Net architecture with Swin Transformer blocks to enhance low-resolution images. This implementation combines the power of diffusion models with transformer-based attention mechanisms for high-quality image super-resolution.

## Overview

This project is a complete from-scratch implementation of ResShift, a diffusion model for single image super-resolution (SISR) that efficiently reduces the number of diffusion steps required by shifting the residual between high-resolution and low-resolution images. The model architecture consists of:

- **Encoder**: 4-stage encoder with residual blocks and time embeddings
- **Bottleneck**: Swin Transformer blocks for global feature modeling
- **Decoder**: 4-stage decoder with skip connections from the encoder
- **Noise Schedule**: ResShift schedule (15 timesteps) for the diffusion process

## Features

- 🎯 **ResShift Implementation**: Complete from-scratch implementation of the ResShift paper
- ⚡ **Efficient Diffusion**: Residual shifting mechanism reduces required diffusion steps
- 🔄 **U-Net Architecture**: Encoder-decoder structure with skip connections
- 🧠 **Swin Transformer**: Window-based attention mechanism in the bottleneck
- ⏱️ **Time Conditioning**: Sinusoidal time embeddings for diffusion timesteps
- 📊 **DIV2K Dataset**: Trained on DIV2K high-quality image dataset
- 📈 **Comprehensive Evaluation**: Metrics include PSNR, SSIM, and LPIPS

## Requirements

- Python >= 3.11
- PyTorch >= 2.9.1
- [uv](https://github.com/astral-sh/uv) (Python package manager)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DiffusionSR
```

### 2. Install uv (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

### 3. Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment and install dependencies
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate

# Install project dependencies
uv pip install -e .
```

Alternatively, you can use uv's sync command:

```bash
uv sync
```

## Dataset Setup

The model expects the DIV2K dataset in the following structure:

```
data/
├── DIV2K_train_HR/          # High-resolution training images
└── DIV2K_train_LR_bicubic/
    └── X4/                   # Low-resolution images (4x downsampled)
```

### Download DIV2K Dataset

1. Download the DIV2K dataset from the [official website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
2. Extract the files to the `data/` directory
3. Ensure the directory structure matches the above

**Note**: Update the paths in `src/data.py` (lines 75-76) to match your dataset location:

```python
train_dataset = SRDataset(
    dir_HR = 'path/to/DIV2K_train_HR',
    dir_LR = 'path/to/DIV2K_train_LR_bicubic/X4',
    scale=4,
    patch_size=256
)
```

## Usage

### Training

To train the model, run:

```bash
python src/train.py
```

The training script will:
- Load the dataset using the `SRDataset` class
- Initialize the `FullUNET` model
- Train using the ResShift noise schedule
- Save training progress and loss values

### Training Configuration

Current training parameters (in `src/train.py`):
- **Batch size**: 4
- **Learning rate**: 1e-4
- **Optimizer**: Adam (betas: 0.9, 0.999)
- **Loss function**: MSE Loss
- **Gradient clipping**: 1.0
- **Training steps**: 150
- **Scale factor**: 4x
- **Patch size**: 256x256

You can modify these parameters directly in `src/train.py` to suit your needs.

### Evaluation

The model performance is evaluated using the following metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher PSNR values indicate better image quality reconstruction.

- **SSIM (Structural Similarity Index Measure)**: Assesses the similarity between two images based on luminance, contrast, and structure. SSIM values range from -1 to 1, with higher values (closer to 1) indicating greater similarity to the ground truth.

- **LPIPS (Learned Perceptual Image Patch Similarity)**: Evaluates perceptual similarity between images using deep network features. Lower LPIPS values indicate images that are more perceptually similar to the reference image.

To run evaluation (once implemented), use:

```bash
python src/test.py
```

## Project Structure

```
DiffusionSR/
├── data/                      # Dataset directory (not tracked in git)
│   ├── DIV2K_train_HR/
│   └── DIV2K_train_LR_bicubic/
├── src/
│   ├── config.py             # Configuration file
│   ├── data.py               # Dataset class and data loading
│   ├── model.py              # U-Net model architecture
│   ├── noiseControl.py       # ResShift noise schedule
│   ├── train.py              # Training script
│   └── test.py               # Testing script (to be implemented)
├── pyproject.toml            # Project dependencies and metadata
├── uv.lock                   # Locked dependency versions
└── README.md                 # This file
```

## Model Architecture

### Encoder
- **Initial Conv**: 3 → 64 channels
- **Stage 1**: 64 → 128 channels, 256×256 → 128×128
- **Stage 2**: 128 → 256 channels, 128×128 → 64×64
- **Stage 3**: 256 → 512 channels, 64×64 → 32×32
- **Stage 4**: 512 channels (no downsampling)

### Bottleneck
- Residual blocks with Swin Transformer blocks
- Window size: 7×7
- Shifted window attention for global context

### Decoder
- **Stage 1**: 512 → 256 channels, 32×32 → 64×64
- **Stage 2**: 256 → 128 channels, 64×64 → 128×128
- **Stage 3**: 128 → 64 channels, 128×128 → 256×256
- **Stage 4**: 64 → 64 channels
- **Final Conv**: 64 → 3 channels (RGB output)

## Key Components

### ResShift Noise Schedule
The model implements the ResShift noise schedule as described in the original paper, defined in `src/noiseControl.py`:
- 15 timesteps (0-14)
- Parameters: `eta1=0.001`, `etaT=0.999`, `p=0.8`
- Efficiently shifts the residual between HR and LR images during the diffusion process

### Time Embeddings
Sinusoidal embeddings are used to condition the model on diffusion timesteps, similar to positional encodings in transformers.

### Data Augmentation
The dataset includes:
- Random cropping (aligned between HR and LR)
- Random horizontal/vertical flips
- Random 180° rotation

## Development

### Adding New Features

1. Model modifications: Edit `src/model.py`
2. Training changes: Modify `src/train.py`
3. Data pipeline: Update `src/data.py`
4. Configuration: Add settings to `src/config.py`

## License

[Add your license here]

## Citation

If you use this code in your research, please cite the original ResShift paper:

```bibtex
@article{yue2023resshift,
  title={ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting},
  author={Yue, Zongsheng and Wang, Jianyi and Loy, Chen Change},
  journal={arXiv preprint arXiv:2307.12348},
  year={2023}
}
```

## Acknowledgments

- **ResShift Authors**: Zongsheng Yue, Jianyi Wang, and Chen Change Loy for their foundational work on efficient diffusion-based super-resolution
- DIV2K dataset providers
- PyTorch community
- Swin Transformer architecture inspiration

