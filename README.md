# Image Inpainting GAN

This repository contains an image inpainting project implemented in PyTorch. The main training and experimentation script is `gan_code.py`, which implements a context-aware generator and a multi-scale discriminator, advanced loss terms (perceptual, adversarial with gradient penalty, feature-matching, and contrastive), a curriculum-based mask generator, AMP training, and uncertainty estimation by Monte‑Carlo sampling.


---

## Highlights (from gan_code.py)
- Generator: ContextAwareGenerator (encoder → bottleneck with self-attention & residuals → contextual-attention layer → decoder).
- Discriminator: MultiScaleDiscriminator (spectral-norm convs, multi-stage features, final scalar output).
- Masking: CurriculumMaskGenerator — progressively increases mask difficulty across epochs.
- Losses:
  - Reconstruction (L1) on masked regions
  - Perceptual loss using VGG19 features
  - Adversarial loss with dynamic adversarial weight (ramp-up)
  - Gradient penalty for stable GAN training
  - Feature matching (L1 between discriminator features)
  - Optional contrastive-style perceptual loss (applied periodically)
- Training details:
  - Mixed precision training (torch.amp + GradScaler)
  - AdamW optimizers for G and D
  - Warmup + Cosine LR schedulers (LinearLR → CosineAnnealingLR via SequentialLR)
  - Gradient clipping
  - Checkpointing & sample visualizations saved to `checkpoints/` and `samples/`
  - Uncertainty estimation via multiple forward passes

---

## Requirements
Minimum packages used in the script:
- Python 3.8+
- torch (CUDA-enabled)
- torchvision
- numpy
- pillow (PIL)
- matplotlib
- tqdm

Note: The code asserts that CUDA is available:
```python
assert torch.cuda.is_available(), "CUDA-enabled GPU required!"
```
So a CUDA GPU is required to run the script as-is. For CPU-only execution, the script would need modification.

Suggested quick install:
```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy pillow matplotlib tqdm
```

---

## Repo layout (as used by `gan_code.py`)
- gan_code.py             -- main training script (contains model, losses, and training loop)
- checkpoints/            -- saved model checkpoints (created during training)
- samples/                -- visualization outputs (created during training)
- data/ or your dataset path (user-specified) -- images used for training/eval

---

## Configuration (Config class in gan_code.py)
Important parameters you may tune:
- img_size: 256
- batch_size: 16
- epochs: 60
- num_images: 1600 (limits dataset images used)
- checkpoint_freq: 5
- warmup_epochs: 10
- curriculum_steps: 30
- grad_clip: 0.5
- g_lr / d_lr: 2e-4 / 1e-4
- lambda_rec, lambda_con, lambda_fm, gp_weight, lambda_adv_start, lambda_adv_end
- adv_rampup_epochs: number of epochs for adversarial weight ramp-up

To change config values edit the `Config` class in `gan_code.py`.

---

## Dataset format & recommended preparation
- The dataset used by the script is a directory of images (JPEG/PNG). The dataset wrapper loads up to `Config.num_images` images.
- Default preprocessing: resize to img_size + 32, random crop to img_size, random flips, color jitter, normalization to [-1, 1].
- Default dataset path in the script (for Colab) is:
  `/content/drive/MyDrive/Colab Notebooks/dataset/val_256`
- Place images directly in the directory (flat structure). Corrupt files will be skipped.

---

## How to run
1. Edit `gan_code.py` to set your `data_root` path (or modify the script to accept a CLI argument).
   - By default:
     data_root = '/content/drive/MyDrive/Colab Notebooks/dataset/val_256'
2. Run the script:
```bash
python gan_code.py
```
Notes:
- The script will create `checkpoints/` and `samples/` directories if they don't exist.
- Checkpoints saved: `checkpoints/epoch_{epoch}.pth` and `checkpoints/best_model.pth`.
- Sample visualizations are saved as `samples/epoch_XXXX.png`.

Recommended for Google Colab:
- Mount Google Drive, set `data_root` to a path in your drive (as in the default).
- Use a GPU runtime and install the required packages.

---

## Checkpoints & Outputs
- Checkpoints include model and optimizer states allowing resuming training.
- Visual outputs include a 5-panel image (masked input, prediction, ground truth, error map, uncertainty map).
- Uncertainty estimation: multiple inferences per sample (`num_uncertainty_passes`) and visualize standard deviation.

---

## Model architecture (brief)
- ResidualBlock: conv downsampling + group norm + GELU + residual shortcut
- SelfAttentionBlock: GroupNorm → MultiheadAttention applied to flattened spatial features
- ContextualAttentionLayer: convolutional matching and attention across spatial features (custom contextual match)
- ContextAwareGenerator: concatenates input RGB and mask channels (4 channels) and outputs a 3-channel inpainted image in [-1,1]
- MultiScaleDiscriminator: several spectral-norm conv downsampling stages producing patch-level features and a single scalar output

---

## Losses & training tricks
- Perceptual loss uses early VGG19 features (vgg.features[:16]) and is frozen.
- Gradient penalty is used (WGAN-GP style) for discriminator regularization.
- Feature-matching loss reduces artifacts by matching intermediate discriminator activations.
- Adversarial weight is ramped from `lambda_adv_start` to `lambda_adv_end` over `adv_rampup_epochs`.
- Contrastive-style VGG-based loss is applied at a configurable frequency (`contrastive_loss_frequency`).
- Mixed precision and gradient clipping are used for stability and speed.

---

## Tips & Troubleshooting
- If you hit OOM:
  - Reduce `Config.batch_size`.
  - Reduce `img_size`.
  - Reduce number of workers or disable persistent_workers.
- If CUDA assertion fails, make sure CUDA drivers and the correct PyTorch/CUDA wheel are installed.
- If you see NaN/Inf values, inspect images for corrupt files, or reduce learning rates.
- The dataset loader caches items — if you modify transforms and want fresh samples, restart the script.
- For CPU-only debugging, remove or comment the CUDA assert and replace device handling appropriately.

---

## Contributing
- Open an issue to describe bugs, feature requests, or desired improvements.
- Fork -> create a feature branch -> open a PR. Include tests or examples for large changes.

---

## License & Citation
- No LICENSE file is included by default in the repository. Add a license file (e.g., MIT or Apache-2.0) at the repo root to make usage terms explicit.

---
