import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss (relu3_3); also returns relu1_2, relu2_2 and
    relu3_3 features for the style loss"""
    STYLE_LAYERS = (3, 8, 15)  # relu1_2, relu2_2, relu3_3 of VGG-16

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features[:16].eval()
        for param in self.features.parameters():
            param.requires_grad = False
        # ImageNet normalisation expected by the pretrained VGG-16 (images in [0, 1])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def multi_features(self, x):
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.STYLE_LAYERS:
                feats.append(x)
        return feats

    def forward(self, pred, target):
        pred_feat = self.features((pred - self.mean) / self.std)
        target_feat = self.features((target - self.mean) / self.std)
        return F.mse_loss(pred_feat, target_feat)


class StrokeConnectivityLoss(nn.Module):
    """Damage-focused topological stroke loss.

    1. Ink maps. The grey levels of the intact image are split into two classes
       by Otsu's threshold; the smaller class is the text, which gives its
       polarity p (+1 for dark text, -1 for light text). With mu0 < mu1 the
       class means, t = (mu0 + mu1) / 2 and delta = max(mu1 - mu0, min_contrast),
       V = sigmoid(sharpness * p * (t - gray) / delta) for both images: a soft
       binarization, differentiable everywhere, whose slope adapts to the
       contrast of the text (V = 0.993 and 0.007 at the two class means).
    2. Soft skeletons of the ink maps, obtained by iterated grayscale erosion and
       opening (Shit et al., clDice, CVPR 2021). Erosion and dilation with a flat
       structuring element are min- and max-pooling: piecewise linear functions,
       differentiable almost everywhere, whose gradient goes to the pixel that
       attains the minimum or maximum, exactly as in a max-pooling layer.
    3. Topology precision (1 - share of the restored skeleton that exceeds the
       intact skeleton outside the intact ink) and sensitivity (share of the
       intact skeleton covered by restored ink, credited up to the intact ink
       level), combined as 1 - their harmonic mean, and computed only in a band
       around the damaged pixels D* (dilated by `band` pixels), where strokes
       have to be reconnected; on the whole image when an image has no damaged
       pixel. For binary maps, both reduce to the measures of clDice; for soft
       maps, both equal 1 when the restored image equals the intact one, so the
       loss is exactly 0 there and never pushes strokes beyond the intact ink.
    A broken stroke lowers the sensitivity, a spurious stroke (e.g. a residual
       scribble) lowers the precision."""

    def __init__(self, iters=10, sharpness=10.0, min_contrast=0.1, band=3, smooth=1.0, levels=256):
        super().__init__()
        self.iters, self.sharpness, self.min_contrast = iters, sharpness, min_contrast
        self.band, self.smooth, self.levels = band, smooth, levels

    @staticmethod
    def erode(x):
        """Grayscale erosion (minimum) over a 3 x 3 cross"""
        p1 = -F.max_pool2d(-x, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-x, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    @staticmethod
    def dilate(x):
        """Grayscale dilation (maximum) over a 3 x 3 square"""
        return F.max_pool2d(x, 3, 1, 1)

    def soft_skeleton(self, x):
        skel = F.relu(x - self.dilate(self.erode(x)))
        for _ in range(self.iters):
            x = self.erode(x)
            delta = F.relu(x - self.dilate(self.erode(x)))
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def text_classes(self, gray):
        """Otsu split of the grey levels of each image (sort-based, deterministic):
        threshold t between the class means, polarity of the smaller class and
        contrast between the class means, each of shape (B, 1, 1, 1)."""
        x = gray.flatten(1).sort(dim=1).values
        n = x.shape[1]
        levels = torch.linspace(0, 1, self.levels, device=x.device, dtype=x.dtype).expand(x.shape[0], -1).contiguous()
        count = torch.searchsorted(x, levels, right=True)              # pixels <= level
        csum = torch.cat([torch.zeros_like(x[:, :1]), x.cumsum(dim=1)], dim=1)
        w0, m0, mt = count / n, csum.gather(1, count) / n, csum[:, -1:] / n
        between = (mt * w0 - m0) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
        k = between.argmax(dim=1, keepdim=True)
        w, m = w0.gather(1, k), m0.gather(1, k)
        mu0, mu1 = m / w.clamp_min(1e-12), (mt - m) / (1 - w).clamp_min(1e-12)
        polarity = torch.where(w <= 0.5, torch.ones_like(w), -torch.ones_like(w))
        shape = (-1, 1, 1, 1)
        return ((mu0 + mu1) / 2).view(shape), polarity.view(shape), (mu1 - mu0).view(shape)

    def ink(self, gray, threshold, polarity, contrast):
        return torch.sigmoid(self.sharpness * polarity * (threshold - gray) / contrast.clamp_min(self.min_contrast))

    def forward(self, pred, target, damage=None):
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred, target = pred.float(), target.float()
            g_r, g_t = pred.mean(dim=1, keepdim=True), target.mean(dim=1, keepdim=True)
            t, p, c = self.text_classes(g_t.detach())
            v_r, v_t = self.ink(g_r, t, p, c), self.ink(g_t, t, p, c)
            s_r, s_t = self.soft_skeleton(v_r), self.soft_skeleton(v_t)
            if damage is None:
                w = torch.ones_like(v_r)
            else:
                w = F.max_pool2d(damage.float(), 2 * self.band + 1, 1, self.band)
                empty = w.flatten(1).sum(dim=1) == 0
                w = torch.where(empty.view(-1, 1, 1, 1), torch.ones_like(w), w)
            dims = (1, 2, 3)
            excess = (w * F.relu(s_r - s_t) * (1 - v_t)).sum(dims)
            t_prec = 1 - excess / ((w * s_r).sum(dims) + self.smooth)
            t_sens = ((w * s_t * torch.minimum(v_r, v_t)).sum(dims) + self.smooth) / ((w * s_t * v_t).sum(dims) + self.smooth)
            return (1.0 - 2.0 * t_prec * t_sens / (t_prec + t_sens)).mean()


class MFSSNetLoss(nn.Module):
    """Comprehensive loss function for MFSSNet model"""
    def __init__(self, alpha_pixel=1.0, alpha_perceptual=0.5, alpha_edge=0.2, alpha_char=0.3, alpha_stroke=0.3,
                 alpha_style=120.0, alpha_seg=0.5, alpha_bnd=0.2, beta_kl=1e-3):
        super().__init__()
        self.alpha_pixel = alpha_pixel
        self.alpha_perceptual = alpha_perceptual
        self.alpha_edge = alpha_edge
        self.alpha_char = alpha_char
        self.alpha_stroke = alpha_stroke
        self.alpha_style = alpha_style
        self.alpha_seg = alpha_seg
        self.alpha_bnd = alpha_bnd
        self.beta_kl = beta_kl

        self.vgg = PerceptualLoss()
        self.stroke_loss = StrokeConnectivityLoss()

        self.register_buffer('sobel_x', torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).float())
        self.register_buffer('sobel_y', torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).float())

    def edge_loss(self, pred, target):
        pred_edges_x = F.conv2d(pred.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred.mean(dim=1, keepdim=True), self.sobel_y, padding=1)
        target_edges_x = F.conv2d(target.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target.mean(dim=1, keepdim=True), self.sobel_y, padding=1)

        return F.l1_loss(pred_edges_x, target_edges_x) + F.l1_loss(pred_edges_y, target_edges_y)

    def character_consistency_loss(self, pred, target):
        patch_size = 16
        unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size // 2)

        pred_patches = unfold(pred)
        target_patches = unfold(target)

        pred_patches_norm = F.normalize(pred_patches, dim=1)
        target_patches_norm = F.normalize(target_patches, dim=1)

        similarity = torch.sum(pred_patches_norm * target_patches_norm, dim=1)
        return 1.0 - similarity.mean()

    # ---- style loss (Gram matrices of VGG-16 relu1_2, relu2_2, relu3_3) ----
    @staticmethod
    def gram(f):
        b, c, h, w = f.shape
        f = f.reshape(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def style_loss(self, pred_feats, target_feats):
        # Gram matrices are computed in float32 (sums over H*W overflow in fp16)
        with torch.autocast(device_type=pred_feats[0].device.type, enabled=False):
            return sum(F.l1_loss(self.gram(p.float()), self.gram(t.float()))
                       for p, t in zip(pred_feats, target_feats))

    # ---- auxiliary losses (training only) ----
    def edge_target(self, target):
        """E*: Sobel gradient magnitude of the intact image, normalised to [0, 1]"""
        gray = target.float().mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.float(), padding=1)
        gy = F.conv2d(gray, self.sobel_y.float(), padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        return mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)

    def boundary_loss(self, boundary_logits, target):
        e = self.edge_target(target)
        losses = [F.binary_cross_entropy_with_logits(
                      b.float(), F.adaptive_max_pool2d(e, b.shape[-2:]))
                  for b in boundary_logits]
        return torch.stack(losses).mean()

    @staticmethod
    def damage_loss(logits, target, smooth=1.0):
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + smooth) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
        return bce + dice.mean()

    @staticmethod
    def kl_loss(mu, logvar):
        mu, logvar = mu.float(), logvar.float()
        return 0.5 * (mu ** 2 + logvar.exp() - logvar - 1).sum(dim=1).mean()

    def forward(self, pred, target, aux=None, damage_target=None):
        # A term whose weight is 0 is not computed.
        pixel_loss = F.l1_loss(pred, target)
        total_loss = self.alpha_pixel * pixel_loss
        logs = {'pixel': pixel_loss.item()}
        pred_feats = self.vgg.multi_features(pred)
        target_feats = self.vgg.multi_features(target)
        terms = {
            'perceptual': (self.alpha_perceptual, lambda: F.mse_loss(pred_feats[-1], target_feats[-1])),  # relu3_3
            'edge': (self.alpha_edge, lambda: self.edge_loss(pred, target)),
            'character': (self.alpha_char, lambda: self.character_consistency_loss(pred, target)),
            'stroke': (self.alpha_stroke, lambda: self.stroke_loss(pred, target, damage_target)),
            'style': (self.alpha_style, lambda: self.style_loss(pred_feats, target_feats)),
        }
        # Auxiliary predictions (training only; not used as inputs)
        if aux is not None:
            if aux.get('boundary_logits'):
                terms['boundary'] = (self.alpha_bnd, lambda: self.boundary_loss(aux['boundary_logits'], target))
            if aux.get('logvar') is not None:
                terms['kl'] = (self.beta_kl, lambda: self.kl_loss(aux['mu'], aux['logvar']))
            if damage_target is not None and aux.get('damage_logits') is not None:
                terms['damage'] = (self.alpha_seg, lambda: self.damage_loss(aux['damage_logits'], damage_target))
        for name, (weight, term) in terms.items():
            if weight > 0:
                value = term()
                total_loss = total_loss + weight * value
                logs[name] = value.item()

        return total_loss, logs
