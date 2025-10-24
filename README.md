# Image Inpainting GAN

A PyTorch-based image inpainting project using Generative Adversarial Networks (GANs). This repository provides code to train, evaluate, and run inference with an image inpainting model (generator + discriminator) for filling missing regions in images. The README below is a draft intended to be tailored to the exact implementation in this repo — I couldn't access the repository contents automatically in this session, so if you provide the file list or let me read the code I will update this README to match exact scripts/flags/architectures.

## Features
- Trainable GAN for image inpainting
- Support for common loss terms (adversarial, reconstruction L1/L2, perceptual, style, TV)
- Checkpoint saving and resume training
- Inference script to inpaint arbitrary images given a mask
- Example training and evaluation commands

## Repository structure (example — update if your repo differs)
- data/               -- dataset download/preparation scripts
- datasets/           -- dataset wrappers / PyTorch Dataset classes
- models/             -- generator and discriminator model definitions
- losses/             -- loss functions (e.g., perceptual, adversarial)
- train.py            -- training loop
- eval.py             -- evaluation script
- inference.py        -- run inpainting on single images / folders
- utils/              -- helper functions (logging, checkpoints, image IO)
- checkpoints/        -- saved models
- results/            -- generated images during training/eval
- notebooks/          -- demo notebooks (optional)
- requirements.txt    -- Python dependencies
- README.md           -- this file

## Installation

1. Clone the repo
   ```bash
   git clone https://github.com/Dakshesh-007/image-_inpainting_gan.git
   cd image-_inpainting_gan
   ```

2. Create a virtual environment and install dependencies
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. (Optional) Install additional packages such as torch, torchvision if not in requirements:
   ```bash
   pip install torch torchvision
   ```

## Quick start

Prepare dataset (example: CelebA, Places2, or your own).
- Place images in a folder structure expected by the dataset wrapper (see datasets/).
- If masks are required, the project may generate random masks on-the-fly or accept a masks/ folder.

Train (example)
```bash
python train.py \
  --data_root ./data/celeba \
  --batch_size 16 \
  --num_epochs 100 \
  --checkpoint_dir ./checkpoints \
  --lr 1e-4 \
  --mask_strategy random
```

Resume training from a checkpoint
```bash
python train.py --resume ./checkpoints/last_checkpoint.pth
```

Evaluate
```bash
python eval.py \
  --checkpoint ./checkpoints/best.pth \
  --data_root ./data/val \
  --output_dir ./results/eval
```

Run inference on a single image
```bash
python inference.py \
  --checkpoint ./checkpoints/best.pth \
  --image ./examples/input.jpg \
  --mask ./examples/mask.png \
  --output ./results/inpainted.png
```

If the repository's scripts use different flags, replace the above with the exact CLI options from your code (I can generate exact commands after inspecting train.py / inference.py).

## Model & Training Details (general)
Generator:
- Typically a U-Net / encoder-decoder with skip connections, or a gated convolutional generator for better hole filling.
Discriminator:
- PatchGAN or multi-scale discriminator that focuses on local realism.
Losses:
- Adversarial loss (GAN)
- Reconstruction loss (L1 or L2) on missing regions
- Perceptual loss (VGG feature space) to improve perceptual quality
- Style loss and Total Variation (TV) loss may be used to enhance texture consistency

Training tips:
- Use learning rate scheduling and gradient clipping if training becomes unstable.
- Start with higher weight on reconstruction loss and slowly increase adversarial loss weight.
- Use mixed precision (AMP) to save memory and speed up training if supported.

## Checkpoints & Logging
- Checkpoints should save both model weights and optimizer/scheduler state for resuming training.
- Log training metrics (losses, PSNR, SSIM) and sample images periodically to track progress.
- Optionally integrate TensorBoard or Weights & Biases for visual tracking.

## Dataset notes
- Common choices for inpainting: CelebA-HQ, Places2, Paris StreetView, ImageNet subsets.
- The repository may support random rectangular masks, irregular masks, or user-supplied masks.
- Ensure images are resized / normalized consistently with the network's expected input.

## Example results
- Include before/after examples in results/ showing masked input, ground truth, and inpainted output.
- Report quantitative metrics like PSNR and SSIM on a held-out validation set.

## Common commands (adapt to actual scripts)
- Preview generated images during training (save to results/iteration_xxx.png)
- Convert checkpoints to CPU-only for sharing:
  ```python
  import torch
  ckpt = torch.load('checkpoint.pth', map_location='cpu')
  torch.save(ckpt, 'checkpoint_cpu.pth')
  ```

## Contributing
- Open an issue describing the feature or bug.
- Fork the repository, create a feature branch, and open a pull request.
- Add tests or example notebooks to demonstrate new features.

## License
- Add your chosen license (MIT, Apache-2.0, etc.) in a LICENSE file. If not present, please add one and indicate the license here.

## Citation
If you use this project in your research, please cite the repository and the underlying paper(s) for the inpainting model you used (e.g., "Generative Image Inpainting with Contextual Attention" or more recent gated-conv / partial conv inpainting papers) — replace with the actual paper name used by the implementation.

---

If you'd like, I can:
- Inspect the repository and generate a README that matches exact file names, CLI flags, and the real model architecture (I attempted to read the repo but couldn't access it from this session). To proceed I can either (a) you can allow me to access the repo, (b) provide the list of files / key scripts (train.py, inference.py, models/), or (c) paste train.py and inference.py here and I will create a tailored README and example commands.
