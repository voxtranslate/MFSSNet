import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features[:16].eval()
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        pred_feat = self.features(pred)
        target_feat = self.features(target)
        return F.mse_loss(pred_feat, target_feat)

class StrokeConnectivityLoss(nn.Module):
    """Loss to ensure stroke connectivity and character structure"""
    def __init__(self):
        super().__init__()
        
        # Morphological operations kernels
        self.register_buffer('dilate_kernel', torch.ones(1, 1, 3, 3))
        self.register_buffer('erode_kernel', torch.ones(1, 1, 3, 3))
    
    def forward(self, pred, target):
        # Convert to grayscale
        pred_gray = pred.mean(dim=1, keepdim=True)
        target_gray = target.mean(dim=1, keepdim=True)
        
        # Compute skeleton/medial axis approximation
        pred_skeleton = self.approximate_skeleton(pred_gray)
        target_skeleton = self.approximate_skeleton(target_gray)
        
        # Loss on skeleton preservation
        skeleton_loss = F.l1_loss(pred_skeleton, target_skeleton)
        
        # Connectivity loss via morphological operations
        pred_connected = self.morphological_closing(pred_gray)
        target_connected = self.morphological_closing(target_gray)
        connectivity_loss = F.l1_loss(pred_connected, target_connected)
        
        return skeleton_loss + connectivity_loss
    
    def approximate_skeleton(self, x):
        """Approximate skeleton via erosion-based thinning"""
        eroded = F.conv2d(x, self.erode_kernel / 9, padding=1)
        skeleton = torch.clamp(x - eroded, 0, 1)
        return skeleton
    
    def morphological_closing(self, x):
        """Morphological closing: dilation followed by erosion"""
        dilated = F.conv2d(x, self.dilate_kernel / 9, padding=1)
        closed = F.conv2d(dilated, self.erode_kernel / 9, padding=1)
        return closed


class MFSSNetLoss(nn.Module):
    """Comprehensive loss function for MFSSNet model"""
    def __init__(self, alpha_pixel=1.0, alpha_perceptual=0.5, alpha_edge=0.2, alpha_char=0.3, alpha_stroke=0.3):
        super().__init__()
        self.alpha_pixel = alpha_pixel
        self.alpha_perceptual = alpha_perceptual
        self.alpha_edge = alpha_edge
        self.alpha_char = alpha_char
        self.alpha_stroke = alpha_stroke
        
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

    def forward(self, pred, target):
        pixel_loss = F.l1_loss(pred, target)
        perceptual_loss = self.vgg(pred, target)
        edge_loss = self.edge_loss(pred, target)
        char_loss = self.character_consistency_loss(pred, target)
        stroke_loss = self.stroke_loss(pred, target)
        
        total_loss = (self.alpha_pixel * pixel_loss + 
                     self.alpha_perceptual * perceptual_loss +
                     self.alpha_edge * edge_loss +
                     self.alpha_char * char_loss +
                     self.alpha_stroke * stroke_loss)
        
        return total_loss, {
            'pixel': pixel_loss.item(),
            'perceptual': perceptual_loss.item(),
            'edge': edge_loss.item(),
            'character': char_loss.item(),
            'stroke': stroke_loss.item()
        }