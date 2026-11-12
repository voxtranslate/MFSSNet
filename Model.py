import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================================
# CA-SSM (Content-Adaptive State-Space Model)
# ============================================================================

def _linear_scan(a, b):
    """Inclusive scan of the first-order linear recurrence

        h_t = a_t * h_{t-1} + b_t,   h_0 = 0,   t = 1..L   (along dim 1)

    for tensors a, b of shape (N, L, D). The pair (a_t, b_t) composes with the
    associative operator (a1, b1) o (a2, b2) = (a1 * a2, a2 * b1 + b2), so the
    prefixes are computed with a work-efficient odd/even parallel scan:
    O(L) work and O(log L) depth, with no custom CUDA kernel.
    """
    L = a.shape[1]
    if L == 1:
        return b
    if L % 2 == 1:  # pad with the identity element (a = 1, b = 0)
        a = F.pad(a, (0, 0, 0, 1), value=1.0)
        b = F.pad(b, (0, 0, 0, 1), value=0.0)
    a_even, a_odd = a[:, 0::2], a[:, 1::2]
    b_even, b_odd = b[:, 0::2], b[:, 1::2]
    # Combine neighbouring pairs (2i, 2i+1) and solve the half-length problem:
    # its solution is the state at every odd position.
    h_odd = _linear_scan(a_even * a_odd, a_odd * b_even + b_odd)
    # State at even positions: h_{2i} = a_{2i} * h_{2i-1} + b_{2i}.
    h_prev = F.pad(h_odd[:, :-1], (0, 0, 1, 0))
    h_even = a_even * h_prev + b_even
    h = torch.stack((h_even, h_odd), dim=2).reshape(a.shape)
    return h[:, :L]


class _SelectiveScan(torch.autograd.Function):
    """Linear recurrence with an adjoint backward pass.

    Forward : h_t = a_t * h_{t-1} + b_t
    Backward: lam_t = dL/dh_t + a_{t+1} * lam_{t+1}   (same scan, reverse time)
              dL/db_t = lam_t,   dL/da_t = lam_t * h_{t-1}
    Only a and h are stored, so memory stays linear in the sequence length.
    """

    @staticmethod
    def forward(ctx, a, b):
        with torch.no_grad():
            h = _linear_scan(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        a, h = ctx.saved_tensors
        a_next = F.pad(a[:, 1:], (0, 0, 0, 1), value=0.0)          # a_{t+1}
        lam = _linear_scan(a_next.flip(1), grad_h.flip(1)).flip(1)
        h_prev = F.pad(h[:, :-1], (0, 0, 1, 0))                    # h_{t-1}
        return lam * h_prev, lam


def selective_scan(a, b):
    return _SelectiveScan.apply(a, b)


class CA_SSM(nn.Module):
    """
    CA-SSM: Content-Adaptive State-Space Model.

    A bidirectional selective state-space recurrence run along the reading
    axis (each row of the feature map is one sequence):

        g_t    = sigmoid(W_g2 GELU(W_g1 u_t))                  content gate
        Delta_t = g_t * softplus(W_dt u_t + b_dt)              gated step
        B_t = W_B u_t,  C_t = W_C u_t,  A^(d) = -exp(A_log^(d))
        Abar_t = exp(Delta_t * A^(d))                          ZOH
        Bbar_t = (1 - Abar_t) / (-A^(d)) * B_t                 ZOH
        h_t^(->) = Abar_t * h_{t-1}^(->) + Bbar_t * u_t        left to right
        h_t^(<-) = Abar_t * h_{t+1}^(<-) + Bbar_t * u_t        right to left
        o_t^(d)  = C_t * h_t^(d) + D^(d) * u_t

    When g_t -> 0, Delta_t -> 0, so Abar_t -> 1 and Bbar_t -> 0: the token
    leaves the state unchanged, and the context gathered on each side of it is
    carried across. The state is diagonal (one scalar per channel and per
    direction), i.e. its dimension is d_inner for each direction.
    """

    def __init__(self, dim, expand=2, d_conv=3,
                 dt_min=1e-3, dt_max=1e-1, a_min=1.0, a_max=16.0):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expand)
        d = self.d_inner

        # Input projection into a content stream u and a modulation stream r
        self.in_proj = nn.Linear(dim, 2 * d)
        # Local 2-D context before the row-wise scans
        self.dwconv = nn.Conv2d(d, d, d_conv, padding=d_conv // 2, groups=d)

        # Content gate g_t (scales the discretisation step)
        self.content_gate = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, d),
            nn.Sigmoid(),
        )
        # Input-dependent step, input matrix B_t and output matrix C_t
        self.dt_proj = nn.Linear(d, d)
        self.x_proj = nn.Linear(d, d)
        self.c_proj = nn.Linear(d, d)

        # Direction-specific decay rates and skip terms: index 0 = forward (->),
        # index 1 = backward (<-). -A is initialised on [a_min, a_max] so that the
        # channels start with memory horizons ranging from a few to ~100 columns.
        a_init = torch.linspace(a_min, a_max, d).log()
        self.A_log = nn.Parameter(torch.stack((a_init, a_init.clone())))
        self.D = nn.Parameter(torch.ones(2, d))

        # Output projection of the concatenated directions
        self.out_proj = nn.Linear(2 * d, dim)

        # Step-size bias initialised so that softplus(bias) ~ U[dt_min, dt_max]
        # on a log scale (inverse softplus).
        dt = torch.exp(torch.rand(d) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def _direction(self, u, delta, Bt, Ct, k, reverse):
        """One scan direction. u, delta, Bt, Ct: (N, W, d_inner)."""
        # The scan runs in at least float32, also under mixed precision
        dtype = torch.promote_types(u.dtype, torch.float32)
        A = -torch.exp(self.A_log[k].to(dtype))                # A < 0
        dA = delta.to(dtype) * A
        a = torch.exp(dA)                                      # Abar_t in (0, 1]
        b = (-torch.expm1(dA) / (-A)) * Bt.to(dtype) * u.to(dtype)  # Bbar_t * u_t
        if reverse:
            a, b = a.flip(1), b.flip(1)
        h = selective_scan(a, b)
        if reverse:
            h = h.flip(1)
        return Ct * h.to(u.dtype) + self.D[k] * u

    def forward(self, x):
        Bsz, C, H, W = x.shape
        residual = x

        u, r = self.in_proj(rearrange(x, 'b c h w -> b h w c')).chunk(2, dim=-1)
        u = F.silu(self.dwconv(rearrange(u, 'b h w d -> b d h w')))

        # Each row is one sequence along the reading axis (state reset per row)
        u = rearrange(u, 'b d h w -> (b h) w d')
        r = rearrange(r, 'b h w d -> (b h) w d')

        g = self.content_gate(u)
        delta = g * F.softplus(self.dt_proj(u))
        Bt = self.x_proj(u)
        Ct = self.c_proj(u)

        o_fwd = self._direction(u, delta, Bt, Ct, 0, reverse=False)
        o_bwd = self._direction(u, delta, Bt, Ct, 1, reverse=True)

        o = torch.cat((o_fwd, o_bwd), dim=-1) * F.silu(torch.cat((r, r), dim=-1))
        y = self.out_proj(o)
        y = rearrange(y, '(b h) w c -> b c h w', b=Bsz, h=H)
        return y + residual


# ============================================================================
# CG-CSWA (Character-Guided Cyclic Shift Window Attention)
# ============================================================================

class CG_CSWA(nn.Module):
    """
    CG-CSWA: Character-Guided Cyclic Shift Window Attention

    Novel Contributions:
    1. Predicts character boundaries to guide attention
    2. Boundary-aware attention bias prevents bleeding across characters
    3. Proper cyclic shifting with attention masking
    4. Text-specific attention mechanism

    This addresses the limitation of generic attention that doesn't understand
    character structure, leading to artifacts in degraded text restoration.
    """
    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Character guidance: boundary map B(x) and boundary-aware attention bias.
        self.boundary_logits = None  # kept for the auxiliary boundary loss
        self.boundary_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1)  # logits of the character-boundary map B(x)
        )
        self.boundary_bias_weight = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # Standard relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        # For attention masking in shifted windows
        self.attn_mask = None

    def _create_attention_mask(self, H, W):
        """Create attention mask for shifted window attention"""
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition mask into windows
        mask_windows = self.window_partition(img_mask.permute(0, 3, 1, 2), self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def _create_boundary_bias(self, boundary_windows):
        """
        Create attention bias that discourages attention across character boundaries
        boundary_windows: (num_windows, window_size^2, 1)
        """
        # Pairwise boundary strength: high values when crossing boundaries
        boundary_diff = torch.abs(
            boundary_windows.unsqueeze(2) - boundary_windows.unsqueeze(1)
        )  # (num_windows, window_size^2, window_size^2, 1)

        # Strong penalty when crossing high boundaries
        boundary_penalty = -10.0 * boundary_diff.squeeze(-1).unsqueeze(1)  # Add head dimension

        return boundary_penalty

    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        # B(x) = sigmoid(logits); during training the logits are supervised by
        # the edge map of the intact image (auxiliary loss, not an input)
        self.boundary_logits = self.boundary_predictor(x)  # B x 1 x H x W
        boundary_map = torch.sigmoid(self.boundary_logits)

        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        boundary_map = F.pad(boundary_map, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x.shape

        # Create or update attention mask for shifted windows (cached per padded size)
        if self.shift_size > 0 and (self.attn_mask is None or getattr(self, '_mask_size', None) != (Hp, Wp)
                                    or self.attn_mask.device != x.device):
            self.attn_mask = self._create_attention_mask(Hp, Wp).to(x.device)
            self._mask_size = (Hp, Wp)

        # Cyclic shift (if needed)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            shifted_boundary = torch.roll(boundary_map, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x
            shifted_boundary = boundary_map

        # Partition into windows
        x_windows = self.window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Partition the boundary map, shifted as the features
        boundary_windows = self.window_partition(shifted_boundary, self.window_size)
        boundary_windows = boundary_windows.view(-1, self.window_size * self.window_size, 1)

        # Create boundary-aware attention bias
        boundary_attn_bias = self._create_boundary_bias(boundary_windows)

        # QKV projection
        qkv = self.qkv(x_windows).reshape(
            -1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn + boundary_attn_bias * self.boundary_bias_weight

        # Apply attention mask for shifted windows
        if self.attn_mask is not None:
            nW = self.attn_mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, self.window_size * self.window_size, self.window_size * self.window_size)
            attn = attn + self.attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, self.window_size * self.window_size, self.window_size * self.window_size)

        # Positions added by the padding (e.g. the 8-row bottleneck map in 16 x 16
        # windows) are excluded from the attention as keys
        if pad_r > 0 or pad_b > 0:
            N = self.window_size * self.window_size
            valid = torch.zeros((1, 1, Hp, Wp), device=x.device)
            valid[:, :, :H, :W] = 1.0
            if self.shift_size > 0:
                valid = torch.roll(valid, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            valid = self.window_partition(valid, self.window_size).reshape(-1, N)   # (nW, N)
            key_mask = (1.0 - valid) * -100.0
            attn = attn.view(B, -1, self.num_heads, N, N) + key_mask[None, :, None, None, :]
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)

        # Apply attention
        x_windows = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x_windows = self.proj(x_windows)

        # Merge windows
        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self.window_reverse(x_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()

        return x + r

    def window_partition(self, x, window_size):
        """Partition into non-overlapping windows"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        """Reverse window partition"""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
        return x


# ============================================================================
# GGCF: Gated Global Context Fusion
# ============================================================================

class GGCF(nn.Module):
    """Gated Global Context Fusion: Uses a gating mechanism for adaptive context fusion."""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2

        self.conv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

        # Global context extraction
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.context_proj = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x_local = F.gelu(x)

        # Extract global context
        gap_feat = self.gap(x_local)
        gmp_feat = self.gmp(x_local)
        global_context = torch.cat([gap_feat, gmp_feat], dim=1)
        global_context = self.context_proj(global_context)

        broadcasted_context = global_context.expand_as(x_local)
        gate_input = torch.cat([x_local, broadcasted_context], dim=1)
        fusion_gate = self.fusion_gate(gate_input)

        x_fused = x_local * (1 - fusion_gate) + broadcasted_context * fusion_gate

        x = self.dwconv(x_fused)
        x = F.gelu(x)
        x = self.conv2(x)
        x = self.dropout(x)

        return x + residual


# ============================================================================
# AFM: Adaptive Feature Masking
# ============================================================================

class AFM(nn.Module):
    """AFM: Adaptive Feature Masking - Feature Masking in Mixer"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.mask_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv(x)
        mask = self.mask_gen(x)
        return feat * mask + x


# ============================================================================
# FDS: Frequency Decomposition Stage
# ============================================================================

class FDS(nn.Module):
    """FDS: Frequency Decomposition Stage - H-L and L-H frequency processing"""
    def __init__(self, dim):
        super().__init__()
        # High to Low frequency transformation
        self.h2l = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 1)
        )

        # Low to High frequency transformation
        self.l2h = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 1)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Extract low frequency (global smooth features)
        x_low = self.gap(x)

        # Extract high frequency (detail features)
        x_high = x - F.interpolate(F.avg_pool2d(x, 3, 1, 1), size=x.shape[2:], mode='bilinear')

        # Cross-frequency enhancement
        h2l = self.h2l(x_high)
        l2h = F.interpolate(self.l2h(x_low), size=x.shape[2:], mode='bilinear')

        return x_high + l2h, x_low.expand_as(x) + h2l


# ============================================================================
# CAM: Channel Attention Module
# ============================================================================

class CAM(nn.Module):
    """CAM: Channel Attention Module for feature recalibration"""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.gap(x))


# ============================================================================
# FFM: Frequency Feature Mixer
# ============================================================================

class FFM(nn.Module):
    """FFM: Frequency Feature Mixer - Feature Mixing of Mixer"""
    def __init__(self, dim):
        super().__init__()
        self.spatial_mix = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.channel_mix = nn.Conv2d(dim, dim, 1)
        self.cam = CAM(dim)

    def forward(self, x_h, x_l):
        # Mix high and low frequency features
        x = self.spatial_mix(x_h + x_l)
        x = self.channel_mix(x)
        x = self.cam(x) * x
        return x + x_h + x_l


# ============================================================================
# SRB: Spatial Restoration Block
# ============================================================================

class SRB(nn.Module):
    """
    SRB: Spatial Restoration Block
    - CA-SSM: Content-Adaptive State-Space Model
    - CG-CSWA: Character-Guided Cyclic Shift Window Attention
    - GGCF: Gated Global Context Fusion
    """
    def __init__(self, dim, num_heads=8, window_size=7, use_shift=False):
        super().__init__()
        # Content-Adaptive SSM for long-range dependencies along the reading axis
        self.ca_ssm = CA_SSM(dim)

        # Character-Guided Cyclic Shift Window Attention
        shift_size = window_size // 2 if use_shift else 0
        self.cg_cswa = CG_CSWA(dim, num_heads, window_size, shift_size)

        # Global Context Fusion
        self.ggcf = GGCF(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x):
        # Long-range with CA-SSM
        x = x + self.ca_ssm(self.apply_norm(x, self.norm1))

        # Character-guided attention with CG-CSWA
        x = x + self.cg_cswa(self.apply_norm(x, self.norm2))

        # Global context fusion
        x = x + self.ggcf(self.apply_norm(x, self.norm3))

        return x

    def apply_norm(self, x, norm):
        x = rearrange(x, 'b c h w -> b h w c')
        x = norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


# ============================================================================
# FRB: Frequency Restoration Block
# ============================================================================

class FRB(nn.Module):
    """FRB: Frequency Restoration Block - AFM + FDS + style FiLM + FFM"""
    def __init__(self, dim, style_dim=None):
        super().__init__()
        self.afm = AFM(dim)
        self.fds = FDS(dim)
        self.ffm = FFM(dim)
        # Style-informed modulation of the high-frequency component (bottleneck
        # and decoder blocks). Zero-initialised, i.e. identity at the start.
        self.film = None
        if style_dim is not None:
            self.film = nn.Linear(style_dim, 2 * dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, z=None):
        r = x
        # Adaptive feature masking
        x_masked = self.afm(x)

        # Frequency decomposition
        x_high, x_low = self.fds(x_masked)

        # Style-informed modulation: h_high <- (1 + gamma(z)) * h_high + beta(z)
        if self.film is not None and z is not None:
            gamma, beta = self.film(z).chunk(2, dim=-1)
            x_high = (1 + gamma[:, :, None, None]) * x_high + beta[:, :, None, None]

        # Frequency feature mixing
        x_freq = self.ffm(x_high, x_low)

        return x_freq + r


# ============================================================================
# HRB: Hierarchical Restoration Block
# ============================================================================

class HRB(nn.Module):
    """HRB: Hierarchical Restoration Block - Dual-path processing"""
    def __init__(self, dim, num_heads=8, window_size=7, use_shift=False, style_dim=None):
        super().__init__()
        # Spatial restoration path
        self.srb = SRB(dim, num_heads, window_size, use_shift)

        # Frequency restoration path (style-modulated when style_dim is given)
        self.frb = FRB(dim, style_dim)

        # Path fusion
        self.fusion = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x, z=None):
        # Spatial restoration: CA-SSM + CG-CSWA + GGCF
        x_spatial = self.srb(x)

        # Frequency restoration: AFM + FDS + (style FiLM) + FFM
        x_freq = self.frb(x, z)

        # Fuse both paths
        x_fused = self.fusion(torch.cat([x_spatial, x_freq], dim=1))

        return x_fused + x


# ============================================================================
# PRT: Patch Refinement Transformer
# ============================================================================

class PRTLayer(nn.Module):
    """PRT Layer: Single Patch Refinement Transformer Layer"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        # Self-attention for patch relationships
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]

        # MLP for patch refinement
        x = x + self.mlp(self.norm2(x))

        return x


class PRT(nn.Module):
    """PRT: Patch Refinement Transformer for fine-grained detail enhancement"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            PRTLayer(dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])

    def forward(self, x):
        r = x
        for layer in self.layers:
            x = layer(x)
        return x + r


# ============================================================================
# DSH: Damage Segmentation Head / RWSE: Reliability-Weighted Style Encoder
# ============================================================================

class DSH(nn.Module):
    """Damage Segmentation Head: logits of the per-pixel damage probability D_hat.
    Supervised during training by the damage map D* computed from each training
    pair (corrupted, intact); it never receives a mask as input."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
        )

    def forward(self, f):
        return self.net(f)


class RWSE(nn.Module):
    """Reliability-Weighted variational Style Encoder.
    Pools the mean and standard deviation of projected features over the pixels
    predicted as intact (weights rho = 1 - D_hat, gradient stopped) and maps them
    to a Gaussian style code z ~ N(mu, sigma^2). z is sampled with the
    reparameterisation trick during training and set to mu at inference."""
    def __init__(self, dim, style_dim=64, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.proj = nn.Conv2d(dim, style_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * style_dim, 2 * style_dim),
            nn.GELU(),
            nn.Linear(2 * style_dim, 2 * style_dim),
        )

    def forward(self, f, damage_logits):
        phi = self.proj(f)
        rho = 1.0 - torch.sigmoid(damage_logits.detach())       # reliability
        w = rho / (rho.sum(dim=(2, 3), keepdim=True) + self.eps)
        mean = (w * phi).sum(dim=(2, 3))
        var = (w * (phi - mean[:, :, None, None]) ** 2).sum(dim=(2, 3))
        std = torch.sqrt(var + self.eps)
        mu, logvar = self.mlp(torch.cat((mean, std), dim=1)).chunk(2, dim=1)
        logvar = logvar.clamp(-10.0, 10.0)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return z, mu, logvar


# ============================================================================
# MFSSNet
# ============================================================================

# Switches stored in the 'model_config' of checkpoints written by the version of
# the code that included the ablation study, with their value for the full model.
FULL_MODEL_SWITCHES = dict(use_cassm=True, use_cgcswa=True, use_ggcf=True, use_prt=True,
                           boundary_guidance=True, use_dsh=True, style_mode='variational',
                           ssm_recurrence=True)


class MFSSNet(nn.Module):
    """
    MFSSNet: A MULTI-LEVEL FLEXIBLE STYLE-INFORMED SEGMENTATION-AWARE PERSPECTIVE

    Novel Contributions:
    1. CA-SSM: Content-Adaptive State-Space Model (bidirectional selective scan)
    2. CG-CSWA: Character-Guided Cyclic Shift Window Attention
    3. HRB: Hierarchical dual-path processing (spatial + frequency)
    4. DSH + RWSE: damage segmentation head and reliability-weighted variational
       style encoder, whose style code modulates the bottleneck and decoder
    5. PRT: Patch Refinement Transformer for final detail enhancement
    """
    def __init__(self, in_channels=3, out_channels=3, dim=64,
                 num_blocks=[2, 2, 2, 2],
                 num_heads=[4, 8, 16, 32],
                 window_sizes=[8, 8, 8, 8],
                 patch_size=4,
                 num_layers=4,
                 style_dim=64):
        super().__init__()

        self.patch_size = patch_size

        # SFE: Shallow Feature Extraction
        self.sfe = nn.Conv2d(in_channels, dim, 3, padding=1)

        # HFE: Hierarchical Feature Extraction (Encoder)
        self.hfe_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        current_dim = dim
        for num, heads, ws in zip(num_blocks[:-1], num_heads[:-1], window_sizes[:-1]):
            blocks = nn.ModuleList([
                HRB(current_dim, heads, ws, use_shift=(j % 2 == 1))
                for j in range(num)
            ])
            self.hfe_blocks.append(blocks)
            self.downsamplers.append(nn.Conv2d(current_dim, current_dim * 2, 3, stride=2, padding=1))
            current_dim = current_dim * 2

        # BNE: Bottleneck Enhancement (style-modulated)
        self.bne = nn.ModuleList([
            HRB(current_dim, num_heads[-1], window_sizes[-1], use_shift=(j % 2 == 1), style_dim=style_dim)
            for j in range(num_blocks[-1])
        ])

        # HFR: Hierarchical Feature Reconstruction (Decoder, style-modulated)
        self.hfr_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()

        for num, heads, ws in zip(reversed(num_blocks[:-1]), reversed(num_heads[:-1]), reversed(window_sizes[:-1])):
            self.upsamplers.append(nn.ConvTranspose2d(current_dim, current_dim // 2, 2, stride=2))
            current_dim = current_dim // 2
            self.skip_fusions.append(nn.Conv2d(current_dim * 2, current_dim, 1))

            blocks = nn.ModuleList([
                HRB(current_dim, heads, ws, use_shift=(j % 2 == 1), style_dim=style_dim)
                for j in range(num)
            ])
            self.hfr_blocks.append(blocks)

        # DSH and RWSE, both applied to the output of the first encoder stage
        self.dsh = DSH(dim)
        self.rwse = RWSE(dim, style_dim)

        # PRT: Patch Refinement Transformer
        patch_dim = patch_size * patch_size * dim
        self.prt = PRT(patch_dim, num_heads=8, mlp_ratio=4., num_layers=num_layers)

        # DFR: Deep Feature Reconstruction
        self.dfr = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 7, padding=3),
            nn.GELU(),
            nn.Conv2d(dim // 2, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

    @classmethod
    def from_model_config(cls, model_config):
        """Builds the model from the 'model_config' stored in a checkpoint. The
        switches written by the ablation version of the code are accepted when
        they describe the full model (the only one defined here)."""
        kwargs = dict(model_config)
        for key, full_value in FULL_MODEL_SWITCHES.items():
            value = kwargs.pop(key, full_value)
            if value != full_value:
                raise ValueError(f"checkpoint of an ablation variant ({key}={value!r}): "
                                 "only the full MFSSNet is defined in this code")
        return cls(**kwargs)

    def forward(self, x, return_aux=False):
        # The only input is the corrupted image: no mask is given at any time.
        # SFE: Shallow Feature Extraction
        feat = self.sfe(x)

        # HFE: Hierarchical Feature Extraction
        skips = []
        for blocks, down in zip(self.hfe_blocks, self.downsamplers):
            for block in blocks:
                feat = block(feat)
            skips.append(feat)
            feat = down(feat)

        # Damage map (logits) and style code from the first encoder stage
        damage_logits = self.dsh(skips[0])
        z, mu, logvar = self.rwse(skips[0], damage_logits)

        # BNE: Bottleneck Enhancement (style-modulated)
        for block in self.bne:
            feat = block(feat, z)

        # HFR: Hierarchical Feature Reconstruction
        for blocks, up, skip_fusion, skip in zip(
            self.hfr_blocks, self.upsamplers, self.skip_fusions, reversed(skips)
        ):
            feat = up(feat)
            feat = skip_fusion(torch.cat([feat, skip], dim=1))
            for block in blocks:
                feat = block(feat, z)

        # PRT: Patch Refinement Transformer on non-overlapping patches
        B, C, H, W = feat.shape
        patch_seq = rearrange(feat, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        refined_seq = self.prt(patch_seq)
        refined_spatial = rearrange(refined_seq, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                    h=H // self.patch_size, w=W // self.patch_size,
                                    p1=self.patch_size, p2=self.patch_size)

        # DFR: Deep Feature Reconstruction
        output = self.dfr(refined_spatial)

        if return_aux:
            aux = {
                'damage_logits': damage_logits,
                'mu': mu,
                'logvar': logvar,
                'boundary_logits': [m.boundary_logits for m in self.modules()
                                    if isinstance(m, CG_CSWA) and m.boundary_logits is not None],
            }
            return output, aux
        return output
