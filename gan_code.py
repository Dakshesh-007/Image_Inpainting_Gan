

# %% Cell 1: Environment Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import time
import warnings

# Filter unnecessary warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# GPU configuration
assert torch.cuda.is_available(), "CUDA-enabled GPU required!"
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler()
print(f"Using device: {torch.cuda.get_device_name(0)}")

# %% Cell 2: Optimized Configuration
class Config:
    # Architecture
    img_size = 256
    min_mask_size = 16

    # Training parameters
    batch_size = 16          # Optimized for Colab T4 memory
    epochs = 60               # Total training epochs (~3 hours)
    num_images = 1600         # Number of images to use
    checkpoint_freq = 5       # Save checkpoints every 5 epochs
    warmup_epochs = 10        # Learning rate warmup period
    curriculum_steps = 30     # Curriculum learning duration

    # Loss weights
    lambda_rec = 1.5          # Reconstruction loss weight
    lambda_adv_start = 0.05   # Initial adversarial loss weight
    lambda_adv_end = 0.2      # Final adversarial loss weight
    lambda_con = 0.4          # Contrastive loss weight
    lambda_fm = 0.1           # Feature matching loss weight
    gp_weight = 5.0           # Gradient penalty weight

    # Optimization
    g_lr = 2e-4               # Generator learning rate
    d_lr = 1e-4               # Discriminator learning rate
    grad_clip = 0.5           # Gradient clipping threshold
    num_workers = 2           # Matches Colab's recommendation

    # Technical
    mask_ratio_range = (0.1, 0.5)  # Mask size range
    num_uncertainty_passes = 5     # Uncertainty estimation passes
    contrastive_loss_frequency = 5 # Contrastive loss application frequency
    d_update_freq = 1              # Discriminator update frequency

    # Newly added parameter for adversarial ramp-up epochs
    adv_rampup_epochs = 10

config = Config()

# %% Cell 3: Curriculum Mask Generator
class CurriculumMaskGenerator:
    def __init__(self):
        self.epoch_progress = 0.0
        self.max_holes = 10
        self.size_jitter = 0.25
        self.current_holes = (1, 2)
        self.current_size_range = (0.1, 0.3)

    def update(self, epoch):
        self.epoch_progress = min(epoch / config.curriculum_steps, 1.0)
        self.current_holes = (
            1 + int(3 * self.epoch_progress),
            3 + int(5 * self.epoch_progress))
        self.current_size_range = (
            config.mask_ratio_range[0] + 0.1 * self.epoch_progress,
            config.mask_ratio_range[1] * (0.5 + 0.5 * self.epoch_progress))

    def generate(self):
        base_mask = torch.ones((1, config.img_size, config.img_size))
        num_holes = random.randint(*self.current_holes)
        for _ in range(num_holes):
            hole_ratio = np.random.uniform(*self.current_size_range)
            hole_w = int(config.img_size * hole_ratio * (1 + np.random.uniform(-self.size_jitter, self.size_jitter)))
            hole_h = int(config.img_size * hole_ratio * (1 + np.random.uniform(-self.size_jitter, self.size_jitter)))
            hole_w = max(config.min_mask_size, hole_w)
            hole_h = max(config.min_mask_size, hole_h)
            x = random.randint(0, config.img_size - hole_w)
            y = random.randint(0, config.img_size - hole_h)
            base_mask[:, y:y+hole_h, x:x+hole_w] = 0
        return base_mask

# %% Cell 4: Robust Dataset Class
class InpaintingDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.image_paths = self._get_valid_files()[:config.num_images]
        print(f"Loaded {len(self.image_paths)} images")
        self.cache = {}
        self.transform = transforms.Compose([
            transforms.Resize(config.img_size + 32),
            transforms.RandomCrop(config.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.masker = CurriculumMaskGenerator()

    def _get_valid_files(self):
        valid_files = []
        for f in os.listdir(self.root):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(self.root, f)
                try:
                    with Image.open(path) as img:
                        img.verify()
                    valid_files.append(path)
                except Exception as e:
                    print(f"Skipping invalid file {path}: {str(e)}")
        return valid_files

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        path = self.image_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
            img = self.transform(img)
            mask = self.masker.generate()
            masked = img * mask

            if torch.isnan(masked).any() or torch.isinf(masked).any():
                raise ValueError("Invalid values in masked image")

            self.cache[idx] = (masked, mask, img)
            return masked, mask, img
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return self[random.randint(0, len(self)-1)]

# %% Cell 5: Generator Network with Consistent Channels
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 2, 1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.GroupNorm(8, out_c)
        )
        self.shortcut = nn.Conv2d(in_c, out_c, 1, 2)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.shortcut(x))

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, dim)
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.size()
        x_norm = self.norm(x)
        x_flat = x_norm.view(B, C, -1).permute(0, 2, 1)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        return x + attn_out.permute(0, 2, 1).view(B, C, H, W)

class ContextualAttentionLayer(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.conv_match = nn.utils.spectral_norm(nn.Conv2d(in_channels, 512, 3, padding=1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        match = self.conv_match(x)
        x_flat = x.view(B, C, -1)
        match_flat = match.view(B, 512, -1)
        scores = torch.bmm(x_flat.permute(0, 2, 1), match_flat) / (C**0.5)
        scores = self.softmax(scores)
        attended = torch.bmm(x_flat, scores.permute(0, 2, 1))
        return attended.view(B, C, H, W)

class ContextAwareGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 5, 2, 2),   # 256x256 → 128x128
            nn.GELU(),
            ResidualBlock(64, 128),       # 128x128 → 64x64
            ResidualBlock(128, 256),      # 64x64 → 32x32
            ResidualBlock(256, 512)       # 32x32 → 16x16
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            SelfAttentionBlock(512),
            ResidualBlock(512, 512),
            SelfAttentionBlock(512)
        )

        # Attention
        self.context_attention = ContextualAttentionLayer(512)

        # Decoder
        self.decoder = nn.Sequential(
            ResidualBlock(512, 256),      # 16x16 → 32x32
            ResidualBlock(256, 128),      # 32x32 → 64x64
            ResidualBlock(128, 64),       # 64x64 → 128x128
            nn.ConvTranspose2d(64, 3, 3, 2, 1, 1),  # 128x128 → 256x256
            nn.Tanh()
        )

    def forward(self, x, mask):
        x_in = torch.cat([x, mask], 1)  # Concatenate along channel dimension
        features = self.encoder(x_in)
        features = self.bottleneck(features)
        features = self.context_attention(features)
        return self.decoder(features).clamp(-1, 1)

# %% Cell 6: Discriminator Network
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, 5, 2, 2)),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.1)
            )
        self.stages = nn.ModuleList([
            block(3, 64),    # 256→128
            block(64, 128),  # 128→64
            block(128, 256), # 64→32
            block(256, 512)  # 32→16
        ])
        self.final = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return self.final(x), features

# %% Cell 7: Loss Function
class InpaintingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.to(device)
        self.mse = nn.MSELoss()

    def perceptual_loss(self, pred, real):
        return self.mse(self.vgg(pred), self.vgg(real.detach()))

    def gradient_penalty(self, D, real, fake):
        alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
        interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_out = D(interpolates)[0]
        gradients = torch.autograd.grad(
            outputs=d_out, inputs=interpolates,
            grad_outputs=torch.ones_like(d_out),
            create_graph=True, retain_graph=True
        )[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def contrastive_loss(self, pred, real, mask):
        masked_pred = (pred * mask).clamp(-1, 1)
        masked_real = (real * mask).clamp(-1, 1)
        return F.mse_loss(self.vgg(masked_pred), self.vgg(masked_real.detach()))

    def dynamic_adv_weight(self, epoch):
        return config.lambda_adv_start + (config.lambda_adv_end - config.lambda_adv_start) * \
               min(epoch / config.adv_rampup_epochs, 1.0)

    def compute_d_loss(self, D, real, fake):
        real_pred, _ = D(real)
        fake_pred, _ = D(fake.detach())
        gp = self.gradient_penalty(D, real, fake)
        loss_D = -real_pred.mean() + fake_pred.mean() + config.gp_weight * gp
        return loss_D, {'gp': gp.item()}

    def forward(self, G, D, real, fake, mask, epoch):
        rec_loss = F.l1_loss(fake * mask, real * mask)
        perc_loss = self.perceptual_loss(fake * mask, real * mask)
        fake_pred, fake_feats = D(fake)
        real_feats = D(real.detach())[1]
        adv_loss = -fake_pred.mean()
        fm_loss = sum(F.l1_loss(f, r.detach()) for f, r in zip(fake_feats, real_feats))

        # Apply contrastive loss with frequency
        con_loss = self.contrastive_loss(fake, real, mask) if (epoch % config.contrastive_loss_frequency == 0) \
            else torch.tensor(0.0, device=device)

        total_loss = (config.lambda_rec * (rec_loss + 0.5 * perc_loss) +
                    self.dynamic_adv_weight(epoch) * adv_loss +
                    config.lambda_fm * fm_loss +
                    config.lambda_con * con_loss)

        return {
            'total': total_loss,
            'rec': rec_loss.item(),
            'perc': perc_loss.item(),
            'adv': adv_loss.item(),
            'fm': fm_loss.item(),
            'con': con_loss.item()
        }

# %% Cell 8: Training System
class Trainer:
    def __init__(self, data_root):
        self.dataset = InpaintingDataset(data_root)
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        self.G = ContextAwareGenerator().to(device)
        self.D = MultiScaleDiscriminator().to(device)
        self.criterion = InpaintingLoss()

        # Optimizers
        self.opt_G = optim.AdamW(self.G.parameters(), lr=config.g_lr, betas=(0.5, 0.999))
        self.opt_D = optim.AdamW(self.D.parameters(), lr=config.d_lr, betas=(0.5, 0.999))

        # Schedulers with explicit end_factor for LinearLR
        warmup_G = LinearLR(self.opt_G, start_factor=0.01, total_iters=config.warmup_epochs, end_factor=1.0)
        cosine_G = CosineAnnealingLR(self.opt_G, T_max=config.epochs - config.warmup_epochs)
        self.scheduler_G = SequentialLR(self.opt_G, schedulers=[warmup_G, cosine_G], milestones=[config.warmup_epochs])

        warmup_D = LinearLR(self.opt_D, start_factor=0.01, total_iters=config.warmup_epochs, end_factor=1.0)
        cosine_D = CosineAnnealingLR(self.opt_D, T_max=config.epochs - config.warmup_epochs)
        self.scheduler_D = SequentialLR(self.opt_D, schedulers=[warmup_D, cosine_D], milestones=[config.warmup_epochs])

        self.best_psnr = 0.0
        self.metrics = {'g_loss': [], 'd_loss': [], 'psnr': [], 'con_loss': []}

    def compute_psnr(self, pred, real):
        mse = F.mse_loss(pred.clamp(-1,1), real.clamp(-1,1))
        return 10 * torch.log10(1.0 / (mse + 1e-8)).item()

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'metrics': self.metrics
        }
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(state, f'checkpoints/epoch_{epoch}.pth')
        if is_best:
            torch.save(state, 'checkpoints/best_model.pth')

    def visualize(self, epoch, masked, pred, real, uncertainty, mask):
        denorm = lambda x: (x * 0.5 + 0.5).clamp(0, 1).cpu().detach().numpy()

        masked_img = denorm(masked[0]).transpose(1, 2, 0)
        pred_img = denorm(pred[0]).transpose(1, 2, 0)
        real_img = denorm(real[0]).transpose(1, 2, 0)
        uncertainty_map = uncertainty[0].squeeze().cpu().detach().numpy()
        mask_overlay = mask[0].cpu().detach().numpy().squeeze() > 0.5

        plt.figure(figsize=(24, 6))
        plt.suptitle(f'Epoch {epoch+1} - PSNR: {self.metrics["psnr"][-1]:.2f} dB')

        # Subplot 1: Masked Input
        plt.subplot(1,5,1)
        plt.imshow(masked_img)
        plt.imshow(np.ma.masked_where(mask_overlay, mask_overlay), cmap='cool', alpha=0.3)
        plt.axis('off')

        # Subplot 2: Prediction
        plt.subplot(1,5,2)
        plt.imshow(pred_img)
        plt.axis('off')

        # Subplot 3: Ground Truth
        plt.subplot(1,5,3)
        plt.imshow(real_img)
        plt.axis('off')

        # Subplot 4: Error Map
        plt.subplot(1,5,4)
        error_map = np.abs(pred_img - real_img).mean(axis=-1)
        plt.imshow(error_map, cmap='hot', vmin=0, vmax=0.5)
        plt.axis('off')

        # Subplot 5: Uncertainty
        plt.subplot(1,5,5)
        plt.imshow(uncertainty_map, cmap='viridis', vmin=0, vmax=1)
        plt.axis('off')

        plt.tight_layout()
        os.makedirs("samples", exist_ok=True)
        plt.savefig(f'samples/epoch_{epoch:04d}.png', bbox_inches='tight', dpi=150)
        plt.close()

    def estimate_uncertainty(self, masked, mask):
        self.G.train()
        with torch.no_grad():
            samples = torch.stack([self.G(masked, mask) for _ in range(config.num_uncertainty_passes)])
        return samples.std(0).mean(1, keepdim=True)

    def train(self):
        os.makedirs("samples", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)
        start_time = time.time()

        try:
            for epoch in range(config.epochs):
                self.dataset.masker.update(epoch)
                g_losses, d_losses, psnrs, con_losses = [], [], [], []

                for masked, mask, real in tqdm(self.loader, desc=f'Epoch {epoch+1}/{config.epochs}'):
                    masked = masked.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    real = real.to(device, non_blocking=True)

                    # --- Discriminator Update ---
                    self.opt_D.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        fake = self.G(masked, mask).detach()
                        loss_D, _ = self.criterion.compute_d_loss(self.D, real, fake)
                    scaler.scale(loss_D).backward()
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), config.grad_clip)
                    scaler.step(self.opt_D)

                    # --- Generator Update ---
                    self.opt_G.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        fake = self.G(masked, mask)
                        loss_dict = self.criterion(self.G, self.D, real, fake, mask, epoch)
                        psnr = self.compute_psnr(fake * mask, real * mask)
                    scaler.scale(loss_dict['total']).backward()
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), config.grad_clip)
                    scaler.step(self.opt_G)
                    scaler.update()

                    g_losses.append(loss_dict['total'].item())
                    d_losses.append(loss_D.item())
                    psnrs.append(psnr)
                    con_losses.append(loss_dict['con'])

                self.scheduler_G.step()
                self.scheduler_D.step()

                self.metrics['g_loss'].append(np.mean(g_losses))
                self.metrics['d_loss'].append(np.mean(d_losses))
                self.metrics['psnr'].append(np.mean(psnrs))
                self.metrics['con_loss'].append(np.mean(con_losses))

                if (epoch + 1) % config.checkpoint_freq == 0:
                    with torch.no_grad():
                        uncertainty = self.estimate_uncertainty(masked, mask)
                    self.visualize(epoch, masked, fake, real, uncertainty, mask)
                    self.save_checkpoint(epoch)

                elapsed = time.time() - start_time
                if elapsed > 3 * 3600:  # 3 hours limit
                    print(f"\nTime limit reached after {epoch+1} epochs")
                    break

                if self.metrics['psnr'][-1] > self.best_psnr:
                    self.best_psnr = self.metrics['psnr'][-1]
                    self.save_checkpoint(epoch, is_best=True)

                print(f"Epoch {epoch+1}/{config.epochs} | "
                      f"G Loss: {self.metrics['g_loss'][-1]:.3f} | "
                      f"D Loss: {self.metrics['d_loss'][-1]:.3f} | "
                      f"PSNR: {self.metrics['psnr'][-1]:.2f} dB")

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving final state...")
            self.save_checkpoint(epoch)
        finally:
            print("\nSaving final model...")
            self.save_checkpoint(config.epochs-1)

# %% Cell 9: Execution
if __name__ == '__main__':
    # Set the dataset path as mounted on Google Drive
    data_root = '/content/drive/MyDrive/Colab Notebooks/dataset/val_256'
    try:
        trainer = Trainer(data_root)
        trainer.train()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        print("Immediate actions:")
        print("1. Verify dataset path is correct")
        print("2. Check images are valid (no corrupt files)")
        print("3. Reduce batch_size if memory errors occur")
