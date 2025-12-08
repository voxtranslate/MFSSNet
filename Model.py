import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# CA-SSM (Content-Adaptive State-Space Model)
# ============================================================================

class CA_SSM(nn.Module):
    """
    CA-SSM: Content-Adaptive State-Space Model with Lightweight Selectivity
    
    Novel Contributions:
    1. Content-adaptive gating for input-dependent filtering
    2. Input-dependent output projection C(x) 
    3. Bidirectional processing for text (forward + backward)
    This addresses the limitation of basic SSMs while avoiding Mamba's complexity.
    """
    def __init__(self, dim, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.dim = dim
        self.d_inner = int(dim * expand)
        self.d_state = d_state
        
        # Input projection
        self.in_proj = nn.Linear(dim, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, 
            padding=d_conv//2, groups=self.d_inner
        )
        
        self.content_gate = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 4),
            nn.GELU(),
            nn.Linear(self.d_inner // 4, self.d_inner),
            nn.Sigmoid()
        )
        
        self.A_log = nn.Parameter(torch.randn(self.d_inner))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Input-dependent B and dt
        self.x_proj = nn.Linear(self.d_inner, self.d_inner)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        self.c_proj = nn.Linear(self.d_inner, self.d_inner)
        
        self.backward_proj = nn.Linear(self.d_inner, self.d_inner)
        
        self.out_proj = nn.Linear(self.d_inner * 2, dim)  # *2 for bidirectional
        
        # Positional encoding for spatial awareness
        self.pe = nn.Parameter(torch.randn(1, dim, 1, 1) * 0.02)
    
    def _process_direction(self, x, reverse=False):
        """Process sequence in one direction with content-adaptive SSM"""
        if reverse:
            x = torch.flip(x, dims=[1])
        
        # Content-adaptive gating (provides lightweight selectivity)
        gate = self.content_gate(x)  # B, L, D - Input-dependent gate
        
        # SSM computation
        dt = F.softplus(self.dt_proj(x))
        A = -torch.exp(self.A_log)
        B = self.x_proj(x)
        C = self.c_proj(x)  # Input-dependent output projection
        
        # State update with gating
        A_scaled = A.unsqueeze(0).unsqueeze(0) * dt
        state = torch.exp(A_scaled) * x + B
        
        # Apply content-adaptive gate (selective filtering)
        state = state * gate
        
        # Output projection with input-dependent C
        output = state * C + self.D.unsqueeze(0).unsqueeze(0) * x
        
        if reverse:
            output = torch.flip(output, dims=[1])
        
        return output
    
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # Add positional encoding
        x = x + self.pe
        
        # Flatten spatial dimensions
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Input projection
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Conv1d for local context
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = rearrange(x, 'b d l -> b l d')
        x = F.silu(x)
        
        # Bidirectional processing (forward + backward)
        x_forward = self._process_direction(x, reverse=False)
        x_backward = self._process_direction(x, reverse=True)
        
        # Combine both directions
        x = torch.cat([x_forward, x_backward], dim=-1)
        
        # Gating with residual
        x = x * F.silu(torch.cat([res, res], dim=-1))
        
        # Output projection
        x = self.out_proj(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x + residual


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
        
        self.boundary_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()  # Probability of character boundary
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
        """Create attention mask for shifted window attention - PROPER IMPLEMENTATION"""
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
        # Compute pairwise boundary strength
        # High values when crossing boundaries between positions
        boundary_diff = torch.abs(
            boundary_windows.unsqueeze(2) - boundary_windows.unsqueeze(1)
        )  # (num_windows, window_size^2, window_size^2, 1)
        
        # Strong penalty when crossing high boundaries
        boundary_penalty = -10.0 * boundary_diff.squeeze(-1).unsqueeze(1)  # Add head dimension
        
        return boundary_penalty
    
    def forward(self, x):
        B, C, H, W = x.shape
        r = x
        boundary_map = self.boundary_predictor(x)  # B x 1 x H x W
        
        # Pad if needed
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        boundary_map = F.pad(boundary_map, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x.shape
        
        # Create or update attention mask for shifted windows
        if self.shift_size > 0 and (self.attn_mask is None or self.attn_mask.shape[1] != Hp // self.window_size * Wp // self.window_size):
            self.attn_mask = self._create_attention_mask(Hp, Wp).to(x.device)
        
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
        
        # Partition boundary map
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
# GCF: Global Context Fusion
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
            nn.Conv2d(dim, dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim, 1)
        )
        
        # Low to High frequency transformation
        self.l2h = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim, 1)
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
# SRB: Spatial Restoration Block (UPDATED with Novel Components)
# ============================================================================

class SRB(nn.Module):
    """
    SRB: Spatial Restoration Block with Novel Components
    - CA-SSM: Content-Adaptive State-Space Model
    - CG-CSWA: Character-Guided Cyclic Shift Window Attention
    - GGCF: Gated Global Context Fusion
    """
    def __init__(self, dim, num_heads=8, window_size=7, use_shift=False):
        super().__init__()
        
        # NOVEL: Content-Adaptive SSM for long-range dependencies
        self.ca_ssm = CA_SSM(dim)
        
        # NOVEL: Character-Guided Cyclic Shift Window Attention
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
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        x = norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


# ============================================================================
# FRB: Frequency Restoration Block
# ============================================================================

class FRB(nn.Module):
    """FRB: Frequency Restoration Block - AFM + FDS + FFM"""
    def __init__(self, dim):
        super().__init__()
        self.afm = AFM(dim)
        self.fds = FDS(dim)
        self.ffm = FFM(dim)
        
    def forward(self, x):
        r = x
        # Adaptive feature masking
        x_masked = self.afm(x)
        
        # Frequency decomposition
        x_high, x_low = self.fds(x_masked)
        
        # Frequency feature mixing
        x_freq = self.ffm(x_high, x_low)
        
        return x_freq + r


# ============================================================================
# HRB: Hierarchical Restoration Block
# ============================================================================

class HRB(nn.Module):
    """HRB: Hierarchical Restoration Block - Dual-path processing"""
    def __init__(self, dim, num_heads=8, window_size=7, use_shift=False):
        super().__init__()
        # Spatial restoration path (with novel components)
        self.srb = SRB(dim, num_heads, window_size, use_shift)
        
        # Frequency restoration path
        self.frb = FRB(dim)
        
        # Path fusion
        self.fusion = nn.Conv2d(dim * 2, dim, 1)
        
    def forward(self, x):
        # Spatial restoration: CA-SSM + CG-CSWA + GGCF
        x_spatial = self.srb(x)
        
        # Frequency restoration: AFM + FDS + FFM
        x_freq = self.frb(x)
        
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
# MFSSNet: Hierarchical Image Restoration Network (UPDATED)
# ============================================================================

class MFSSNet(nn.Module):
    """
    MFSSNet: A MULTI-LEVEL FLEXIBLE STYLE-INFORMED SEGMENTATION-AWARE PERSPECTIVE
    
    Novel Contributions:
    1. CA-SSM: Content-Adaptive State-Space Model with lightweight selectivity
    2. CG-CSWA: Character-Guided Cyclic Shift Window Attention
    3. HRB: Hierarchical dual-path processing (spatial + frequency)
    4. PRT: Patch Refinement Transformer for final detail enhancement
    """
    def __init__(self, in_channels=3, out_channels=3, dim=64, 
                 num_blocks=[2, 2, 2, 2], 
                 num_heads=[4, 8, 16, 32],
                 window_sizes=[8, 8, 8, 8],
                 patch_size=4,
                 num_layers=4):
        super().__init__()
        
        self.patch_size = patch_size
        
        # SFE: Shallow Feature Extraction
        self.sfe = nn.Conv2d(in_channels, dim, 3, padding=1)
        
        # HFE: Hierarchical Feature Extraction (Encoder)
        self.hfe_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        current_dim = dim
        for stage_idx, (num, heads, ws) in enumerate(zip(num_blocks[:-1], num_heads[:-1], window_sizes[:-1])):
            blocks = nn.ModuleList([
                HRB(current_dim, heads, ws, use_shift=(j % 2 == 1)) 
                for j in range(num)
            ])
            self.hfe_blocks.append(blocks)
            self.downsamplers.append(nn.Conv2d(current_dim, current_dim*2, 3, stride=2, padding=1))
            current_dim = current_dim * 2
        
        # BNE: Bottleneck Enhancement
        self.bne = nn.ModuleList([
            HRB(current_dim, num_heads[-1], window_sizes[-1], use_shift=(j % 2 == 1))
            for j in range(num_blocks[-1])
        ])
        
        # HFR: Hierarchical Feature Reconstruction (Decoder)
        self.hfr_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        
        for stage_idx, (num, heads, ws) in enumerate(zip(
            reversed(num_blocks[:-1]), 
            reversed(num_heads[:-1]), 
            reversed(window_sizes[:-1])
        )):
            self.upsamplers.append(nn.ConvTranspose2d(current_dim, current_dim//2, 2, stride=2))
            current_dim = current_dim // 2
            self.skip_fusions.append(nn.Conv2d(current_dim*2, current_dim, 1))
            
            blocks = nn.ModuleList([
                HRB(current_dim, heads, ws, use_shift=(j % 2 == 1))
                for j in range(num)
            ])
            self.hfr_blocks.append(blocks)
        
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
        
    def forward(self, x):
        # SFE: Shallow Feature Extraction
        shallow_feat = self.sfe(x)
        
        # HFE: Hierarchical Feature Extraction
        feat = shallow_feat
        skips = []
        for blocks, down in zip(self.hfe_blocks, self.downsamplers):
            for block in blocks:
                feat = block(feat)
            skips.append(feat)
            feat = down(feat)
        
        # BNE: Bottleneck Enhancement
        for block in self.bne:
            feat = block(feat)
        
        # HFR: Hierarchical Feature Reconstruction
        for blocks, up, skip_fusion, skip in zip(
            self.hfr_blocks, self.upsamplers, self.skip_fusions, reversed(skips)
        ):
            feat = up(feat)
            feat = skip_fusion(torch.cat([feat, skip], dim=1))
            for block in blocks:
                feat = block(feat)

        patches = feat
        # Reshape to patches
        B, C, H, W = patches.shape
        patch_seq = rearrange(patches, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # PRT: Patch Refinement Transformer
        refined_seq = self.prt(patch_seq)
        
        # Sequence to patch conversion
        refined_spatial = rearrange(refined_seq, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=H//self.patch_size, w=W//self.patch_size, 
                                   p1=self.patch_size, p2=self.patch_size)
        
        # DFR: Deep Feature Reconstruction
        output = self.dfr(refined_spatial)
        
        return output