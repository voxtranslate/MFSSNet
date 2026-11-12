import csv
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from Model import MFSSNet  # repo-import
from Loss import MFSSNetLoss  # repo-import
from Dataset import build_benchmark_datasets, set_seed, split_manifest  # repo-import

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for MFSSNet training and evaluation"""
    def __init__(self):
        # Model architecture
        self.in_channels = 3
        self.out_channels = 3
        self.dim = 32
        self.num_blocks = [2, 2, 2, 2]
        self.num_heads = [2, 4, 8, 16]
        self.window_sizes = [8, 8, 16, 16]
        self.patch_size = 4
        self.num_prt_layers = 4
        self.style_dim = 64        # dimension of the style code z

        # Training (these values are the ones used; they are reported in Section 4.1)
        self.batch_size = 52
        self.epochs = 30
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.img_size = (64, 256)

        # Mixed Precision & Gradient Clipping
        self.use_amp = True  # Automatic Mixed Precision
        self.grad_clip_norm = 0.5  # Gradient clipping max norm

        # Advanced Scheduler Configuration
        self.warmup_epochs = 5  # Linear warmup epochs
        self.scheduler_t0 = 50  # CosineAnnealing: epochs until first restart
        self.scheduler_tmult = 2  # CosineAnnealing: period multiplier after restart

        # Loss weights
        self.alpha_pixel = 1.0
        self.alpha_perceptual = 0.5
        self.alpha_edge = 0.3
        self.alpha_char = 0.3
        self.alpha_stroke = 0.3
        self.alpha_style = 120.0   # Gram-matrix style loss
        self.alpha_seg = 0.5       # damage segmentation (BCE + Dice)
        self.alpha_bnd = 0.2       # boundary map supervision
        self.beta_kl = 1e-3        # KL of the variational style code
        self.damage_tau = 0.1      # threshold of the damage target D*

        # Benchmark protocol (Zhu et al., AAAI 2024)
        self.seed = 42
        self.train_on = 'ht'          # 'st' or 'ht': one model per benchmark
        self.val_fraction = 0.02      # validation split held out from the TRAINING set
        self.noise_prob = 0.0         # 0 = no augmentation (same training data as the baselines)
        self.test_noise_sigma = 0.0   # 0 = test images used as distributed
        self.strict_counts = True     # stop if a split differs from the official one
        self.check_leakage = True     # check that no test image is a training image
        # Model selection: highest validation PSNR, a criterion that does not
        # depend on the loss weights
        self.selection_metric = 'psnr'

        # Every source folder of the official splits must be listed
        # (TII-ST: synthetic + IC13 + IC15 + IC17; TII-HT: IAM)
        self.st_train_dirs = ['/kaggle/working/stt_train']
        self.st_test_dirs = ['/kaggle/working/stt_test']
        self.ht_train_dirs = ['/kaggle/working/htt_train']
        self.ht_test_dirs = ['/kaggle/working/htt_test']

        # Evaluation
        self.metrics = ['psnr', 'ssim']   # validation metrics (selection on 'psnr')

        # Paths
        self.checkpoint_dir = '/content'
        self.results_dir = './results/MFSSNet'
        self.analysis_dir = './results/analysis'
        self.manipulated_dir = './results/manipulated'

        # Resume training
        self.resume = True

    def create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.checkpoint_dir,
            self.results_dir,
            self.analysis_dir,
            f"{self.manipulated_dir}/corrupted",
            f"{self.manipulated_dir}/restored",
            f"{self.manipulated_dir}/clean",
            f"{self.analysis_dir}/comprehensive",
            f"{self.analysis_dir}/lam_maps"
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


def model_kwargs(config):
    """Arguments of MFSSNet for a config."""
    return dict(in_channels=config.in_channels, out_channels=config.out_channels, dim=config.dim,
                num_blocks=config.num_blocks, num_heads=config.num_heads,
                window_sizes=config.window_sizes, patch_size=config.patch_size,
                num_layers=config.num_prt_layers, style_dim=config.style_dim)


# ============================================================================
# METRICS
# ============================================================================

class MetricsEvaluator:
    """Centralized metrics evaluation"""
    def __init__(self, metrics=None, device='cuda'):
        self.device = device
        self.metrics_names = metrics if metrics else ['psnr', 'ssim']

        # Try to import pyiqa if available
        try:
            import pyiqa
            self.metrics = {}
            for metric_name in self.metrics_names:
                try:
                    self.metrics[metric_name] = pyiqa.create_metric(metric_name, device=device)
                except Exception as e:
                    logger.warning(f"Failed to initialize {metric_name}: {e}")
        except ImportError:
            logger.warning("pyiqa not available, using basic metrics")
            self.metrics = {}

        if not self.metrics:
            logger.warning("Using fallback PSNR/SSIM computation")

    def compute_psnr(self, pred, target):
        """Compute PSNR"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(100.0)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def compute_ssim(self, pred, target):
        """Simplified SSIM computation"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, 3, 1, 1)
        mu_target = F.avg_pool2d(target, 3, 1, 1)

        sigma_pred = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_pred * mu_pred
        sigma_target = F.avg_pool2d(target * target, 3, 1, 1) - mu_target * mu_target
        sigma_pred_target = F.avg_pool2d(pred * target, 3, 1, 1) - mu_pred * mu_target

        ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2))

        return ssim_map.mean()

    def compute(self, pred, target):
        # Metrics in float32: under autocast pred is float16, and pyiqa's range check
        # (x.max() < 1 + 1e-6) rounds 1 + 1e-6 to 1.0 in float16, so a pixel equal to
        # 1.0 was rejected and no PSNR/SSIM was computed.
        pred = torch.clamp(pred.float(), 0, 1)
        target = torch.clamp(target.float(), 0, 1)

        results = {}

        if self.metrics:
            for name, metric_fn in self.metrics.items():
                try:
                    if name in ['niqe', 'brisque', 'musiq']:
                        score = metric_fn(pred)
                    else:
                        score = metric_fn(pred, target)
                    results[name] = score.mean().item() if torch.is_tensor(score) else score
                except Exception as e:
                    logger.warning(f"Error computing {name}: {e}")
        else:
            # Fallback to basic metrics
            results['psnr'] = self.compute_psnr(pred, target).item()
            results['ssim'] = self.compute_ssim(pred, target).item()

        return results


# ============================================================================
# BENCHMARK EVALUATION (protocol and metric code of Zhu et al., AAAI 2024)
# ============================================================================
# - official test sets used as distributed (64x256), no added noise
# - restored images saved as 8-bit PNG under their original file names, so that
#   the official recognizers (CRNN, ASTER, MORAN / DAN, TrOCR-B, TrOCR-L) can be
#   run on them exactly as on the outputs of the compared methods
# - PSNR on RGB in [0, 255] and SSIM on RGB (11x11 Gaussian window, sigma 1.5),
#   computed from the saved PNG files and averaged over images, as in the
#   metric code released with GSDM (metrics/psnr.py, metrics/ssim.py);
#   files are paired by name.

def _gsdm_gaussian_window(window_size=11, sigma=1.5, channel=3):
    g = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                      for x in range(window_size)])
    g = (g / g.sum()).unsqueeze(1)
    w2d = g.mm(g.t()).float().unsqueeze(0).unsqueeze(0)
    return w2d.expand(channel, 1, window_size, window_size).contiguous()


def gsdm_psnr(a, b):
    """PSNR of GSDM's metrics/psnr.py (inputs in [0, 255]); returns 0 when the
    images are identical, as the reference code does."""
    mse = torch.mean((a.float() - b.float()) ** 2)
    if mse == 0:
        return 0.0
    return (20 * math.log10(255.0) - 10 * torch.log10(mse)).item()


def gsdm_ssim(img1, img2, window_size=11):
    """SSIM of GSDM's metrics/ssim.py (inputs in [0, 1], RGB, mean of the map)."""
    channel = img1.shape[1]
    w = _gsdm_gaussian_window(window_size, 1.5, channel).to(img1.device, img1.dtype)
    p = window_size // 2
    mu1 = F.conv2d(img1, w, padding=p, groups=channel)
    mu2 = F.conv2d(img2, w, padding=p, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    s1 = F.conv2d(img1 * img1, w, padding=p, groups=channel) - mu1_sq
    s2 = F.conv2d(img2 * img2, w, padding=p, groups=channel) - mu2_sq
    s12 = F.conv2d(img1 * img2, w, padding=p, groups=channel) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return ssim_map.mean().item()


def _to_png(t):
    """[0,1] tensor (3,H,W) -> 8-bit PIL image"""
    arr = (t.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr)


@torch.no_grad()
def evaluate_benchmark(model, test_set, out_dir, device, batch_size=16, groups_csv=None):
    """Restores every image of an official test set, saves the outputs and
    computes PSNR / SSIM with the metric code of the benchmark.
    groups_csv (optional): CSV with columns name,group (e.g. corrosion pattern
    or ratio range) to report the metrics per group."""
    out_dir = Path(out_dir)
    model.eval()
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    for batch in loader:
        restored = torch.clamp(model(batch['corrupted'].to(device)), 0.0, 1.0).cpu()
        for i in range(restored.shape[0]):
            d = out_dir / batch['source'][i]
            d.mkdir(parents=True, exist_ok=True)
            _to_png(restored[i]).save(d / batch['name'][i])

    to_t = transforms.ToTensor()
    rows = []
    for pair in test_set.image_pairs:
        gt = Image.open(pair['clean']).convert('RGB')
        out = Image.open(out_dir / pair['source'] / pair['name']).convert('RGB')
        if out.size != gt.size:
            out = out.resize(gt.size, Image.BICUBIC)
        g, o = to_t(gt).unsqueeze(0), to_t(out).unsqueeze(0)
        rows.append({'source': pair['source'], 'name': pair['name'],
                     'psnr': gsdm_psnr(g * 255., o * 255.), 'ssim': gsdm_ssim(g, o)})

    with open(out_dir / 'per_image_metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'name', 'psnr', 'ssim'])
        w.writeheader()
        w.writerows(rows)

    summary = {'n_images': len(rows),
               'psnr': float(np.mean([r['psnr'] for r in rows])),
               'ssim': float(np.mean([r['ssim'] for r in rows]))}
    if groups_csv:
        with open(groups_csv) as f:
            group_of = {r['name']: r['group'] for r in csv.DictReader(f)}
        by_group = defaultdict(list)
        for r in rows:
            if r['name'] in group_of:
                by_group[group_of[r['name']]].append(r)
        summary['groups'] = {k: {'n_images': len(v),
                                 'psnr': float(np.mean([r['psnr'] for r in v])),
                                 'ssim': float(np.mean([r['ssim'] for r in v]))}
                             for k, v in sorted(by_group.items())}
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


@torch.no_grad()
def time_inference(model, test_set, device, n_images=200, warmup=20):
    """Inference time per image (batch size 1), after warm-up; to be run for all
    timed methods on the same GPU."""
    model.eval()
    times = []
    n = min(len(test_set), n_images + warmup)
    for k in range(n):
        x = test_set[k]['corrupted'].unsqueeze(0).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if k >= warmup:
            times.append(time.perf_counter() - t0)
    gpu = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    return {'device': gpu, 'n_images': len(times),
            'mean_s': float(np.mean(times)), 'std_s': float(np.std(times))}


def mask_iou(pred, target):
    """IoU of two boolean masks (per image); None when both are empty."""
    union = (pred | target).sum()
    if union == 0:
        return None
    return float((pred & target).sum() / union)


@torch.no_grad()
def evaluate_damage_head(model, test_set, device, batch_size=16, out_csv=None):
    """Coverage of the damage and mask IoU of the damage head.
    Target: damage map D* of each (corrupted, intact) test pair (Eq. (19), tau = 0.1).
    Prediction: DSH probability > 0.5, from the corrupted image alone.
    Both are averaged over images; images whose prediction and target are both
    empty are left out of the IoU mean and counted separately. out_csv (optional):
    per-image values (source, name, coverage, iou)."""
    model.eval()
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    cov, ious, n_empty, rows = [], [], 0, []
    for batch in loader:
        _, aux = model(batch['corrupted'].to(device), return_aux=True)
        pred = (aux['damage_logits'] > 0).cpu()
        target = batch['damage'] > 0.5
        for i in range(pred.shape[0]):
            cov.append(float(target[i].float().mean()))
            v = mask_iou(pred[i], target[i])
            if v is None:
                n_empty += 1
            else:
                ious.append(v)
            rows.append({'source': batch['source'][i], 'name': batch['name'][i],
                         'coverage': cov[-1], 'iou': '' if v is None else v})
    if out_csv is not None:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['source', 'name', 'coverage', 'iou'])
            w.writeheader()
            w.writerows(rows)
    return {'n_images': len(cov), 'coverage_mean': float(np.mean(cov)), 'coverage_std': float(np.std(cov)),
            'iou_mean': float(np.mean(ious)) if ious else None, 'iou_std': float(np.std(ious)) if ious else None,
            'n_iou': len(ious), 'n_empty': n_empty}


# ============================================================================
# TRAINER WITH MIXED PRECISION
# ============================================================================

class MFSSNetTrainer:
    """Trainer for MFSSNet model with Mixed Precision Training"""
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.metrics_evaluator = MetricsEvaluator(
            metrics=config.metrics if hasattr(config, 'metrics') else ['psnr', 'ssim'],
            device=device
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay
        )

        # SCHEDULER: linear warm-up, then cosine annealing with warm restarts, updated
        # at every iteration.
        self.warmup_epochs = config.warmup_epochs if hasattr(config, 'warmup_epochs') else 5
        self.warmup_steps = self.warmup_epochs * len(train_loader)
        self.current_step = 0

        # Main scheduler: Cosine Annealing with Warm Restarts (cycles automatically)
        self.cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.scheduler_t0 if hasattr(config, 'scheduler_t0') else 50,  # First restart at 50 epochs
            T_mult=config.scheduler_tmult if hasattr(config, 'scheduler_tmult') else 2,  # Double period after each restart
            eta_min=config.lr * 0.01  # Minimum LR is 1% of initial LR
        )
        # The warm-up starts from a learning rate of zero (the cosine schedule keeps
        # config.lr as its base value)
        if self.warmup_steps > 0:
            for group in self.optimizer.param_groups:
                group['lr'] = 0.0

        # Model selection uses the validation PSNR, which does not depend on the
        # loss weights (unlike the validation loss)
        self.selection_metric = getattr(config, 'selection_metric', 'psnr')

        # MIXED PRECISION: Initialize GradScaler
        self.use_amp = config.use_amp if hasattr(config, 'use_amp') else True
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping norm
        self.grad_clip_norm = config.grad_clip_norm if hasattr(config, 'grad_clip_norm') else 1.0

        self.criterion = MFSSNetLoss(
            alpha_pixel=config.alpha_pixel,
            alpha_perceptual=config.alpha_perceptual,
            alpha_edge=config.alpha_edge,
            alpha_char=config.alpha_char,
            alpha_stroke=config.alpha_stroke,
            alpha_style=getattr(config, 'alpha_style', 120.0),
            alpha_seg=getattr(config, 'alpha_seg', 0.5),
            alpha_bnd=getattr(config, 'alpha_bnd', 0.2),
            beta_kl=getattr(config, 'beta_kl', 1e-3)
        ).to(self.device)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_score = float('-inf')
        self.train_losses = defaultdict(list)
        self.val_losses = defaultdict(list)
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.results_dir = Path(config.results_dir)

        config.create_directories()

        logger.info(f"Mixed Precision Training: {'ENABLED' if self.use_amp else 'DISABLED'}")
        logger.info(f"Gradient Clipping Norm: {self.grad_clip_norm}")
        logger.info(f"Scheduler: Warmup({self.warmup_epochs}E) + CosineAnnealingWarmRestarts")

    def get_lr(self):
        """Get current learning rate with warmup"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.config.lr * (self.current_step / self.warmup_steps)
        else:
            # Use cosine scheduler LR
            return self.optimizer.param_groups[0]['lr']

    def advance_schedule(self):
        """Learning rate of the next iteration: linear warm-up, then cosine
        annealing with warm restarts evaluated at the number of epochs elapsed
        since the start of training."""
        self.current_step += 1
        self.update_lr()
        if self.current_step >= self.warmup_steps:
            self.cosine_scheduler.step(self.current_step / len(self.train_loader))

    def update_lr(self):
        """Update learning rate with warmup logic"""
        if self.current_step < self.warmup_steps:
            # Warmup phase: linear increase
            lr = self.config.lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        # After warmup, cosine scheduler handles LR updates

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cosine_scheduler_state_dict': self.cosine_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # Save scaler state
            'best_val_loss': self.best_val_loss,
            'best_val_score': self.best_val_score,
            'selection_metric': self.selection_metric,
            'current_step': self.current_step,
            'train_losses': dict(self.train_losses),
            'val_losses': dict(self.val_losses),
            'model_architecture': 'MFSSNet (CA-SSM + CG-CSWA + DSH + RWSE)',
            'model_config': model_kwargs(self.config),   # architecture
            'variant': 'full',
            'train_on': getattr(self.config, 'train_on', None),
            'seed': getattr(self.config, 'seed', None),
            'use_amp': self.use_amp,
            'grad_clip_norm': self.grad_clip_norm,
            'scheduler_type': 'Warmup + CosineAnnealingWarmRestarts'
        }

        # Atomic writes: the file is written under a temporary name and renamed only
        # once complete, so an interrupted save never leaves a truncated latest.pth
        def atomic_save(path):
            tmp = path.with_name(path.name + '.tmp')
            torch.save(checkpoint, tmp)
            os.replace(tmp, path)

        atomic_save(self.checkpoint_dir / 'latest.pth')
        # The loss curves are also kept as a small text file beside the checkpoints, so that the
        # validation curves can be rebuilt without the weights.
        try:
            hist_path = self.checkpoint_dir / 'history.json'
            prev = {}
            if hist_path.exists():
                try:
                    prev = json.loads(hist_path.read_text(encoding='utf-8')).get('epochs', {})
                except (ValueError, OSError):
                    prev = {}
            for part, losses in (('train', self.train_losses), ('val', self.val_losses)):
                for k, vals in dict(losses).items():
                    for j, x in enumerate(vals):
                        prev.setdefault(str(self.current_epoch - len(vals) + 1 + j), {}).setdefault(part, {})[k] = float(x)
            hist_path.write_text(json.dumps({'epoch': self.current_epoch, 'best_val_score': float(self.best_val_score),
                                             'selection_metric': self.selection_metric, 'epochs': prev}), encoding='utf-8')
        except Exception as e:                                           # the history file never stops a training run
            logger.warning(f"history.json not written ({type(e).__name__}: {e})")

        if is_best:
            atomic_save(self.checkpoint_dir / 'best.pth')
            logger.info(f"Saved best model at epoch {self.current_epoch}")

        if self.current_epoch % 25 == 0:
            atomic_save(self.checkpoint_dir / f'epoch_{self.current_epoch}.pth')

    def load_checkpoint(self, path=None):
        if path is None:
            path = self.checkpoint_dir / 'latest.pth'

        if not Path(path).exists():
            logger.info(f"No checkpoint at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler states
        if 'cosine_scheduler_state_dict' in checkpoint:
            self.cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])

        # Load scaler state if available
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_step = checkpoint.get('current_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_score = checkpoint.get('best_val_score', float('-inf'))
        # the loss histories continue across a resume
        for name in ('train_losses', 'val_losses'):
            hist = getattr(self, name)
            for k, vals in (checkpoint.get(name) or {}).items():
                hist[k] = list(vals)

        logger.info(f"Resumed MFSSNet training from epoch {self.current_epoch}")

    def train_epoch(self):
        self.model.train()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(list)

        pbar = tqdm(self.train_loader, desc=f'MFSSNet Epoch {self.current_epoch+1}', leave=False)

        for batch in pbar:
            corrupted = batch['corrupted'].to(self.device)
            clean = batch['clean'].to(self.device)
            damage = batch['damage'].to(self.device)

            self.optimizer.zero_grad()

            # MIXED PRECISION: Forward pass with autocast
            with autocast('cuda', enabled=self.use_amp):
                pred, aux = self.model(corrupted, return_aux=True)
                total_loss, loss_dict = self.criterion(pred, clean, aux=aux, damage_target=damage)

            # MIXED PRECISION: Backward pass with scaler
            self.scaler.scale(total_loss).backward()

            # GRADIENT CLIPPING: Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            # MIXED PRECISION: Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # UPDATE LEARNING RATE (for the next iteration)
            self.advance_schedule()

            # Compute metrics (without autocast for accuracy)
            with torch.no_grad():
                metrics = self.metrics_evaluator.compute(pred, clean)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

            epoch_losses['total'] += total_loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] += v

            current_lr = self.get_lr()
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'LR': f"{current_lr:.6f}",
                'Scale': f"{self.scaler.get_scale():.0f}",
                'Phase': 'Warmup' if self.current_step < self.warmup_steps else 'Cosine'
            })

        num_batches = len(self.train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            self.train_losses[k].append(epoch_losses[k])

        for k, v in epoch_metrics.items():
            if v:
                avg = sum(v) / len(v)
                self.train_metrics[k].append(avg)

        return epoch_losses, {k: sum(v)/len(v) for k, v in epoch_metrics.items() if v}

    def validate_epoch(self):
        self.model.eval()
        epoch_losses = defaultdict(float)
        epoch_metrics = defaultdict(list)

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='MFSSNet Validation', leave=False)

            for batch in pbar:
                corrupted = batch['corrupted'].to(self.device)
                clean = batch['clean'].to(self.device)
                damage = batch['damage'].to(self.device)

                # Use autocast for validation too (eval mode: z = mu, deterministic)
                with autocast('cuda', enabled=self.use_amp):
                    pred, aux = self.model(corrupted, return_aux=True)
                    total_loss, loss_dict = self.criterion(pred, clean, aux=aux, damage_target=damage)

                metrics = self.metrics_evaluator.compute(pred, clean)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

                epoch_losses['total'] += total_loss.item()
                for k, v in loss_dict.items():
                    epoch_losses[k] += v

        num_batches = len(self.val_loader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            self.val_losses[k].append(epoch_losses[k])

        for k, v in epoch_metrics.items():
            if v:
                avg = sum(v) / len(v)
                self.val_metrics[k].append(avg)

        return epoch_losses, {k: sum(v)/len(v) for k, v in epoch_metrics.items() if v}

    def train(self, epochs):
        logger.info(f"Starting MFSSNet training for {epochs} epochs")
        logger.info(f"Model: MFSSNet with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info("Novel Components: CA-SSM + CG-CSWA + DSH + RWSE")
        logger.info(f"Mixed Precision: {'ENABLED' if self.use_amp else 'DISABLED'}")
        logger.info(f"Gradient Clipping: {self.grad_clip_norm}")
        logger.info(f"Scheduler: Warmup({self.warmup_epochs}E) + CosineAnnealingWarmRestarts")

        for epoch in range(self.current_epoch, epochs):
            train_losses, train_metrics = self.train_epoch()
            val_losses, val_metrics = self.validate_epoch()

            # Selection criterion (validation PSNR)
            if self.selection_metric not in val_metrics:
                raise RuntimeError(
                    f"validation metric {self.selection_metric!r} could not be computed on any batch "
                    f"(got {sorted(val_metrics)}); see the 'Error computing' warnings above")
            score = val_metrics[self.selection_metric]

            current_lr = self.get_lr()
            log_msg = (f"Epoch {epoch+1}/{epochs} - "
                      f"LR: {current_lr:.6f} - "
                      f"Train Loss: {train_losses['total']:.4f} - "
                      f"Val Loss: {val_losses['total']:.4f}")

            for k in ['psnr', 'ssim']:
                if k in train_metrics and k in val_metrics:
                    log_msg += f" - Train {k.upper()}: {train_metrics[k]:.3f}"
                    log_msg += f" - Val {k.upper()}: {val_metrics[k]:.3f}"

            logger.info(log_msg)

            self.best_val_loss = min(self.best_val_loss, val_losses['total'])
            is_best = score > self.best_val_score
            if is_best:
                self.best_val_score = score
                logger.info(f"New best validation {self.selection_metric.upper()}: {score:.4f}")

            self.current_epoch = epoch + 1
            self.save_checkpoint(is_best)


# ============================================================================
# INFERENCER
# ============================================================================

class MFSSNetInferencer:
    """Inferencer for MFSSNet model with gradient attribution maps and multiple analysis tools"""
    def __init__(self, model, device, config, checkpoint_path=None):
        self.device = device
        self.model = model.to(device)
        self.config = config
        self.metrics_evaluator = MetricsEvaluator(
            metrics=config.metrics if hasattr(config, 'metrics') else ['psnr', 'ssim'],
            device=device
        )

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded MFSSNet model from {path}")

    def compute_lam(self, corrupted, restored, clean):
        """Gradient attribution map: absolute gradient of the squared error between the
        output and the intact image with respect to the input pixels, averaged over the
        channels and normalized to [0, 1] (Fig. 5(k)). This is a plain gradient saliency
        map, not the path-integrated local attribution map (LAM) of Gu and Dong."""
        corrupted_input = corrupted.clone().detach().requires_grad_(True)

        # We need to forward pass again with grad enabled
        output = self.model(corrupted_input)

        # Loss for backprop (MSE with ground truth)
        loss = F.mse_loss(output, clean)

        self.model.zero_grad()
        loss.backward()

        lam = torch.abs(corrupted_input.grad)
        lam = lam.mean(dim=1, keepdim=True)
        lam = (lam - lam.min()) / (lam.max() - lam.min() + 1e-8)

        return lam.detach()

    def visualize_spectral_density(self, image):
        """Log-magnitude spectrum of the centred 2-D Fourier transform of the grey-level
        image (Fig. 5(h)-(j))."""
        img_np = image.mean(dim=0).cpu().numpy()
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        return magnitude_spectrum

    def visualize_pixel_distribution(self, ax, image, label, color):
        """Visualize histogram of pixel intensities"""
        img_np = image.cpu().numpy().flatten()
        ax.hist(img_np, bins=50, color=color, alpha=0.5, label=label, density=True)

    def _edges(self, restored, clean):
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float().to(self.device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float().to(self.device)

        restored_gray = restored.mean(dim=0, keepdim=True).unsqueeze(0)
        clean_gray = clean.mean(dim=0, keepdim=True).unsqueeze(0)

        restored_edges = torch.sqrt(
            F.conv2d(restored_gray, sobel_x, padding=1)**2 +
            F.conv2d(restored_gray, sobel_y, padding=1)**2
        )
        clean_edges = torch.sqrt(
            F.conv2d(clean_gray, sobel_x, padding=1)**2 +
            F.conv2d(clean_gray, sobel_y, padding=1)**2
        )
        return restored_edges, clean_edges

    def save_individual_components(self, corrupted, restored, clean, image_id, save_dir):
        """Save individual image components and analysis maps"""
        individual_dir = Path(save_dir) / 'individual_components' / f'{image_id}'
        individual_dir.mkdir(parents=True, exist_ok=True)

        to_pil = transforms.ToPILImage()

        # 1. Save basic images
        to_pil(torch.clamp(corrupted, 0.0, 1.0)).save(individual_dir / f'{image_id}_01_corrupted.png')
        to_pil(torch.clamp(restored, 0.0, 1.0)).save(individual_dir / f'{image_id}_02_restored.png')
        to_pil(torch.clamp(clean, 0.0, 1.0)).save(individual_dir / f'{image_id}_03_ground_truth.png')

        # 2. Save difference map
        diff_np = torch.abs(restored - clean).mean(dim=0).cpu().numpy()
        diff_colored = plt.cm.hot(diff_np / (diff_np.max() + 1e-8))[:, :, :3]
        Image.fromarray((diff_colored * 255).astype(np.uint8)).save(individual_dir / f'{image_id}_04_difference_map.png')

        # 3. Save attribution map
        with torch.enable_grad():
            lam = self.compute_lam(corrupted.unsqueeze(0), restored.unsqueeze(0), clean.unsqueeze(0))

        lam_np = lam.squeeze().cpu().numpy()
        lam_colored = plt.cm.jet(lam_np)[:, :, :3]
        Image.fromarray((lam_colored * 255).astype(np.uint8)).save(individual_dir / f'{image_id}_05_lam.png')

        # 4. Save attribution overlay
        corrupted_np = torch.clamp(corrupted, 0.0, 1.0).cpu().permute(1, 2, 0).numpy()
        overlay = np.clip(0.6 * corrupted_np + 0.4 * lam_colored, 0.0, 1.0)
        Image.fromarray((overlay * 255).astype(np.uint8)).save(individual_dir / f'{image_id}_06_lam_overlay.png')

        # 5. Save high attribution regions (binary mask)
        lam_binary = (lam_np > lam_np.mean()).astype(np.uint8) * 255
        Image.fromarray(lam_binary, mode='L').save(individual_dir / f'{image_id}_07_high_attribution.png')

        # 6. Save edge maps
        restored_edges, clean_edges = self._edges(restored, clean)

        restored_edges_np = restored_edges.squeeze().cpu().numpy()
        restored_edges_np = (restored_edges_np / (restored_edges_np.max() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(restored_edges_np, mode='L').save(individual_dir / f'{image_id}_08_restored_edges.png')

        clean_edges_np = clean_edges.squeeze().cpu().numpy()
        clean_edges_np = (clean_edges_np / (clean_edges_np.max() + 1e-8) * 255).astype(np.uint8)
        Image.fromarray(clean_edges_np, mode='L').save(individual_dir / f'{image_id}_09_ground_truth_edges.png')

        # 7. Save edge difference map
        edge_diff_np = torch.abs(restored_edges - clean_edges).squeeze().cpu().numpy()
        edge_diff_colored = plt.cm.hot(edge_diff_np / (edge_diff_np.max() + 1e-8))[:, :, :3]
        Image.fromarray((edge_diff_colored * 255).astype(np.uint8)).save(individual_dir / f'{image_id}_10_edge_difference.png')

        logger.info(f"Saved individual components to {individual_dir}")

    def visualize_lam(self, corrupted, restored, clean, save_path):
        """Visualize the gradient attribution map"""
        with torch.enable_grad():
            lam = self.compute_lam(corrupted.unsqueeze(0), restored.unsqueeze(0), clean.unsqueeze(0))

        corrupted_np = torch.clamp(corrupted, 0.0, 1.0).cpu().permute(1, 2, 0).numpy()
        lam_np = lam.squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(corrupted_np)
        axes[0].set_title('Corrupted Input', fontsize=16, fontweight='bold')
        axes[0].axis('off')

        im1 = axes[1].imshow(lam_np, cmap='jet', interpolation='bilinear')
        axes[1].set_title('Gradient Attribution Map', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        heatmap = plt.cm.jet(lam_np)[:, :, :3]
        overlay = np.clip(0.6 * corrupted_np + 0.4 * heatmap, 0.0, 1.0)
        axes[2].imshow(overlay)
        axes[2].set_title('Attribution Overlay', fontsize=16, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_comprehensive_analysis(self, corrupted, restored, clean, image_id, save_dir):
        """Comprehensive analysis with attribution map, FFT, histograms and edge maps in one file."""
        to_pil = transforms.ToPILImage()
        corrupted_img = to_pil(torch.clamp(corrupted, 0.0, 1.0))
        restored_img = to_pil(torch.clamp(restored, 0.0, 1.0))
        clean_img = to_pil(torch.clamp(clean, 0.0, 1.0))

        with torch.no_grad():
            metrics = self.metrics_evaluator.compute(restored.unsqueeze(0), clean.unsqueeze(0))

        with torch.enable_grad():
            lam = self.compute_lam(corrupted.unsqueeze(0), restored.unsqueeze(0), clean.unsqueeze(0))
        lam_np = lam.squeeze().cpu().numpy()

        # Grid Setup: 4 Rows x 4 Columns
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)

        # --- Row 1: Visual Comparison ---
        for col, (img, title) in enumerate(((corrupted_img, 'Corrupted'), (restored_img, 'Restored'),
                                            (clean_img, 'Ground Truth'))):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')

        diff = torch.abs(restored - clean).mean(dim=0)
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(diff.cpu().numpy(), cmap='hot', interpolation='bilinear')
        ax4.set_title('Error Heatmap', fontsize=14, fontweight='bold')
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # --- Row 2: Attribution map ---
        ax5 = fig.add_subplot(gs[1, 0])
        im5 = ax5.imshow(lam_np, cmap='jet', interpolation='bilinear')
        ax5.set_title('Gradient Attribution Map', fontsize=14, fontweight='bold')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        corrupted_np = torch.clamp(corrupted, 0.0, 1.0).cpu().permute(1, 2, 0).numpy()
        overlay = np.clip(0.6 * corrupted_np + 0.4 * plt.cm.jet(lam_np)[:, :, :3], 0.0, 1.0)

        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(overlay)
        ax6.set_title('Attribution on Input', fontsize=14, fontweight='bold')
        ax6.axis('off')

        # Metrics Display
        ax7 = fig.add_subplot(gs[1, 2:])
        metric_text = '\n'.join([f"{k.upper()}: {v:.4f}" for k, v in metrics.items()])
        ax7.text(0.5, 0.5, metric_text, ha='center', va='center', fontsize=18,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1.0),
                 family='monospace')
        ax7.set_title('Quality Metrics', fontsize=14, fontweight='bold')
        ax7.axis('off')

        # --- Row 3: Frequency Analysis (FFT) ---
        for col, (img, title) in enumerate(((corrupted, 'FFT Corrupted'), (restored, 'FFT Restored'),
                                            (clean, 'FFT Ground Truth'))):
            ax = fig.add_subplot(gs[2, col])
            ax.imshow(self.visualize_spectral_density(img), cmap='viridis')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')

        # Histogram
        ax11 = fig.add_subplot(gs[2, 3])
        self.visualize_pixel_distribution(ax11, restored, 'Restored', 'blue')
        self.visualize_pixel_distribution(ax11, clean, 'GT', 'green')
        ax11.set_title('Pixel Histogram', fontsize=14, fontweight='bold')
        ax11.legend()

        # --- Row 4: Edge Maps ---
        restored_edges, clean_edges = self._edges(restored, clean)

        ax12 = fig.add_subplot(gs[3, 0])
        ax12.imshow(restored_edges.squeeze().cpu().numpy(), cmap='gray')
        ax12.set_title('Restored Edges', fontsize=14, fontweight='bold')
        ax12.axis('off')

        ax13 = fig.add_subplot(gs[3, 1])
        ax13.imshow(clean_edges.squeeze().cpu().numpy(), cmap='gray')
        ax13.set_title('Ground Truth Edges', fontsize=14, fontweight='bold')
        ax13.axis('off')

        ax14 = fig.add_subplot(gs[3, 2])
        ax14.imshow(torch.abs(restored_edges - clean_edges).squeeze().cpu().numpy(), cmap='hot')
        ax14.set_title('Edge Difference', fontsize=14, fontweight='bold')
        ax14.axis('off')

        plt.suptitle(f'MFSSNet Analysis - {image_id}', fontsize=22, fontweight='bold', y=0.98)

        save_path = Path(save_dir) / 'comprehensive' / f'analysis_{image_id}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return metrics

    def infer_image_set(self, image_paths, save_dir=None):
        """Infer a set of (corrupted, ground truth) image pairs, with the full analysis of each."""
        save_dir = Path(self.config.analysis_dir) / 'custom_inference' if save_dir is None else Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'comprehensive').mkdir(exist_ok=True)
        (save_dir / 'lam_maps').mkdir(exist_ok=True)
        (save_dir / 'individual_components').mkdir(exist_ok=True)

        transform = transforms.Compose([transforms.ToTensor()])

        all_results = []

        # The main loop is no_grad; gradients are enabled locally for the attribution maps.
        with torch.no_grad():
            for idx, (corrupted_path, clean_path) in enumerate(tqdm(image_paths, desc='Processing images')):
                try:
                    corrupted = transform(Image.open(corrupted_path).convert('RGB')).unsqueeze(0).to(self.device)
                    clean = transform(Image.open(clean_path).convert('RGB')).unsqueeze(0).to(self.device)

                    restored = torch.clamp(self.model(corrupted), 0.0, 1.0)

                    image_id = f'{idx:03d}_{Path(corrupted_path).stem}'

                    metrics = self.create_comprehensive_analysis(
                        corrupted.squeeze(0), restored.squeeze(0), clean.squeeze(0), image_id, save_dir)

                    self.visualize_lam(corrupted.squeeze(0), restored.squeeze(0), clean.squeeze(0),
                                       save_dir / 'lam_maps' / f'lam_{image_id}.png')

                    self.save_individual_components(
                        corrupted.squeeze(0), restored.squeeze(0), clean.squeeze(0), image_id, save_dir)

                    all_results.append({
                        'index': idx,
                        'image_id': image_id,
                        'corrupted_path': str(corrupted_path),
                        'clean_path': str(clean_path),
                        'metrics': metrics,
                        'components_dir': str(save_dir / 'individual_components' / image_id)
                    })

                    logger.info(f"Processed {idx+1}/{len(image_paths)}: {image_id}")
                    logger.info("  Metrics: " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

                except Exception as e:
                    logger.error(f"Error processing {corrupted_path}: {str(e)}")
                    continue

        if all_results:
            logger.info("\n" + "="*80)
            logger.info("MFSSNet Custom Image Set Inference Summary")
            logger.info("="*80)
            logger.info(f"Total images processed: {len(all_results)}")

            all_metrics = defaultdict(list)
            for result in all_results:
                for k, v in result['metrics'].items():
                    all_metrics[k].append(v)

            final_metrics = {}
            for k, values in all_metrics.items():
                final_metrics[k] = {'mean': np.mean(values), 'std': np.std(values),
                                    'min': np.min(values), 'max': np.max(values)}
                logger.info(f"{k.upper():>12}: {np.mean(values):>7.4f} ± {np.std(values):.4f}")
            logger.info("="*80)

            summary = {'final_metrics': final_metrics, 'num_samples': len(all_results),
                       'results': all_results, 'model': 'MFSSNet'}
            with open(save_dir / 'custom_inference_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)

        return all_results

    def batch_inference(self, data_loader, num_samples=100):
        """Batch inference using a DataLoader, with the full analysis of every sample processed."""
        self.config.create_directories()

        output_dir = Path(self.config.analysis_dir)
        all_metrics = defaultdict(list)
        count = 0
        to_pil = transforms.ToPILImage()
        manipulated_dir = Path(self.config.manipulated_dir)

        with torch.no_grad():
            pbar = tqdm(data_loader, desc='MFSSNet Inference')

            for batch in pbar:
                if num_samples is not None and count >= num_samples:
                    break

                corrupted = batch['corrupted'].to(self.device)
                clean = batch['clean'].to(self.device)
                image_ids = batch['image_id']

                restored = torch.clamp(self.model(corrupted), 0.0, 1.0)

                for i in range(min(corrupted.size(0), num_samples - count if num_samples else corrupted.size(0))):
                    metrics = self.create_comprehensive_analysis(
                        corrupted[i], restored[i], clean[i], image_ids[i], output_dir)

                    self.visualize_lam(corrupted[i], restored[i], clean[i],
                                       output_dir / 'lam_maps' / f'lam_{image_ids[i]}.png')

                    self.save_individual_components(corrupted[i], restored[i], clean[i], image_ids[i], output_dir)

                    to_pil(torch.clamp(corrupted[i].cpu(), 0.0, 1.0)).save(manipulated_dir / 'corrupted' / f"{image_ids[i]}.png")
                    to_pil(torch.clamp(restored[i].cpu(), 0.0, 1.0)).save(manipulated_dir / 'restored' / f"{image_ids[i]}.png")
                    to_pil(torch.clamp(clean[i].cpu(), 0.0, 1.0)).save(manipulated_dir / 'clean' / f"{image_ids[i]}.png")

                    for k, v in metrics.items():
                        all_metrics[k].append(v)

                    count += 1

                if all_metrics:
                    pbar.set_postfix({k: f"{sum(v)/len(v):.3f}" for k, v in all_metrics.items()})

        final_metrics = {k: {'mean': sum(v) / len(v), 'std': np.std(v), 'min': min(v), 'max': max(v)}
                         for k, v in all_metrics.items() if v}

        logger.info("=== MFSSNet Evaluation Results ===")
        for k, stats in final_metrics.items():
            logger.info(f"{k.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        with open(output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump({'final_metrics': final_metrics, 'num_samples': count, 'model': 'MFSSNet'},
                      f, indent=2, default=str)

        return final_metrics


# ============================================================================
# PROFILING HELPER
# ============================================================================

def profile_model_and_components(model, config, device):
    print("\n" + "="*85)
    print("MODEL PROFILING: Parameters, Size, GFLOPS, Inference Speed")
    print("="*85)

    try:
        from thop import profile as thop_profile
        has_thop = True
    except ImportError:
        has_thop = False
        print("WARNING: 'thop' library not found. FLOPs calculation skipped.")
        print("Run 'pip install thop' to enable GFLOPS calculations.\n")

    def run_profiling(name, module, dummy_inputs):
        if module is None:
            return

        module.eval()
        module.to(device)

        if isinstance(dummy_inputs, torch.Tensor):
            dummy_inputs = (dummy_inputs,)
        dummy_inputs = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in dummy_inputs)

        # 1. Parameters & Size
        params = sum(p.numel() for p in module.parameters())
        size_mb = params * 4 / (1024 ** 2)

        # 2. FLOPs
        flops_str = "N/A"
        if has_thop:
            try:
                macs, _ = thop_profile(module, inputs=dummy_inputs, verbose=False)
                flops_str = f"{(macs * 2) / (10 ** 9):.3f} GFLOPS"
            except Exception:
                flops_str = "Error/Unsupported"

        # 3. Inference Speed (Pure)
        warmup, iters, avg_time = 10, 50, 0
        try:
            with torch.no_grad():
                for _ in range(warmup):
                    module(*dummy_inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(iters):
                    module(*dummy_inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                avg_time = (time.perf_counter() - start_time) / iters
        except Exception:
            pass

        time_str = f"{avg_time*1000:.2f} ms" if avg_time > 0 else "Error"
        print(f"{name:<25} | Params: {params/1e6:>7.3f}M | Size: {size_mb:>7.2f}MB | {flops_str:>15} | Speed: {time_str:>10} / pass")

    B, C, H, W = 1, config.in_channels, config.img_size[0], config.img_size[1]
    dummy_img = torch.randn(B, C, H, W)

    run_profiling("Full MFSSNet", model, (dummy_img,))
    print("-" * 85)
    print("Component Profiling:")

    run_profiling("1. SFE", model.sfe, (dummy_img,))

    hfe_in = torch.randn(B, config.dim, H, W)
    damage_in = torch.zeros(B, 1, H, W)
    if len(model.hfe_blocks) > 0 and len(model.hfe_blocks[0]) > 0:
        run_profiling("2. HFE Block (Stage 1)", model.hfe_blocks[0][0], (hfe_in,))

    n_down = len(config.num_blocks) - 1
    bn_dim = config.dim * (2 ** n_down)
    bn_h, bn_w = H // (2 ** n_down), W // (2 ** n_down)
    bn_in = torch.randn(B, bn_dim, bn_h, bn_w)
    z_dummy = torch.randn(B, config.style_dim)

    if len(model.bne) > 0:
        run_profiling("3. BNE Block", model.bne[0], (bn_in, z_dummy))

    hfr_in = torch.randn(B, bn_dim // 2, bn_h * 2, bn_w * 2)
    if len(model.hfr_blocks) > 0 and len(model.hfr_blocks[0]) > 0:
        run_profiling("4. HFR Block (Stage 1)", model.hfr_blocks[0][0], (hfr_in, z_dummy))

    run_profiling("5. DSH", model.dsh, (hfe_in,))
    run_profiling("6. RWSE", model.rwse, (hfe_in, damage_in))

    patch_dim = config.patch_size * config.patch_size * config.dim
    seq_len = (H // config.patch_size) * (W // config.patch_size)
    run_profiling("7. PRT", model.prt, (torch.randn(B, seq_len, patch_dim),))

    run_profiling("8. DFR", model.dfr, (hfe_in,))

    print("-" * 85)
    print("Inner Sub-components (based on Stage 1 shapes):")
    if len(model.hfe_blocks) > 0 and len(model.hfe_blocks[0]) > 0:
        hrb = model.hfe_blocks[0][0]
        run_profiling("- CA-SSM", hrb.srb.ca_ssm, (hfe_in,))
        run_profiling("- CG-CSWA", hrb.srb.cg_cswa, (hfe_in,))
        run_profiling("- GGCF", hrb.srb.ggcf, (hfe_in,))
        run_profiling("- FRB Full", hrb.frb, (hfe_in,))
        run_profiling("  - AFM", hrb.frb.afm, (hfe_in,))
        run_profiling("  - FDS", hrb.frb.fds, (hfe_in,))
        run_profiling("  - FFM", hrb.frb.ffm, (hfe_in, hfe_in))

    print("="*85 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training and evaluation function for MFSSNet"""
    config = Config()   # all the settings of the paper are those of Config (Section 4.1)
    config.resume = True

    # Benchmark protocol: one model per benchmark, trained on its official
    # training set only, selected on a validation split of that training set,
    # and evaluated once on the untouched official test set.
    config.train_on = 'st'  # run a second time with 'ht'
    config.checkpoint_dir = f'./runnings/{config.train_on}'
    set_seed(config.seed)

    config.create_directories()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Model: MFSSNet with CA-SSM + CG-CSWA + DSH + RWSE")
    print(f"Mixed Precision: {'ENABLED' if config.use_amp else 'DISABLED'}")
    print(f"Gradient Clipping: {config.grad_clip_norm}")
    print(f"Scheduler: Warmup({config.warmup_epochs}E) + CosineAnnealingWarmRestarts(T0={config.scheduler_t0}, Tmult={config.scheduler_tmult})")

    # Datasets: official training set (split into train / validation) and official test set
    train_dataset, val_dataset, test_dataset = build_benchmark_datasets(config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[{config.train_on}] Train: {len(train_dataset)}, Validation (from the training set): {len(val_dataset)}, "
          f"Test (official, never used for selection): {len(test_dataset)}")
    # Exact splits of this run (image lists and SHA-256 fingerprints), to be published
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(config.checkpoint_dir) / f'split_{config.train_on}.json', 'w') as f:
        json.dump(split_manifest(train_dataset, val_dataset, test_dataset), f)

    model = MFSSNet(**model_kwargs(config))
    print(f"MFSSNet initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    profile_model_and_components(model, config, device)

    trainer = MFSSNetTrainer(model, train_loader, val_loader, device, config)

    if config.resume:
        trainer.load_checkpoint()

    trainer.train(config.epochs)

    print("Starting MFSSNet inference...")
    inferencer = MFSSNetInferencer(model, device, config, checkpoint_path=trainer.checkpoint_dir / 'best.pth')

    # Benchmark evaluation on the FULL official test set (metric code of the benchmark)
    test_dir = Path(config.results_dir) / f'test_{config.train_on}'
    final_metrics = evaluate_benchmark(model, test_dataset, test_dir, device)
    final_metrics['timing'] = time_inference(model, test_dataset, device)
    final_metrics['damage_head'] = evaluate_damage_head(model, test_dataset, device,
                                                        out_csv=test_dir / 'damage_head.csv')
    with open(test_dir / 'summary.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(final_metrics)

    # Qualitative analysis (attribution maps, figures) on part of the test set; not used for the tables
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    inferencer.batch_inference(test_loader, num_samples=600)

    # Custom image pairs for the qualitative figures (corrupted, intact)
    st_ids = [f'{i:04d}' for i in list(range(0, 1)) + list(range(4, 28)) + list(range(62, 108))
              + [367, 537, 564, 565, 599, 733, 734]]
    st_pairs = [(f'/kaggle/working/stt_test/t_incomplete/{i}.png', f'/kaggle/working/stt_test/t_f/{i}.png')
                for i in st_ids]
    ht_ids = [f'{i:05d}' for i in list(range(0, 26)) + list(range(1000, 1010))]
    ht_pairs = [(f'/kaggle/working/htt_test/Corrupted_Image/{i}.png', f'/kaggle/working/htt_test/Intact_Image/{i}.png')
                for i in ht_ids]
    ht_pairs_2 = [(f'/kaggle/working/htt_test/Corrupted_Image/{i:05d}.png',
                   f'/kaggle/working/htt_test/Intact_Image/{i:05d}.png') for i in range(1509, 1521)]
    image_pairs = ht_pairs_2   # or st_pairs / ht_pairs

    results = inferencer.infer_image_set(image_paths=image_pairs, save_dir='htt_results')

    for result in results:
        print(f"Image {result['index']}: PSNR = {result['metrics']['psnr']:.2f}")


if __name__ == '__main__':
    main()
