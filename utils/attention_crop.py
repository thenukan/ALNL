"""
Attention-based cropping using DeiT vision transformer
"""
import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from PIL import Image
import numpy as np


class AttentionCropper:
    """
    Extract attention-based crops from images using a pre-trained vision transformer
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.feature_extractor = None
        self._init_model()
        
    def _init_model(self):
        """Initialize DeiT model for attention extraction"""
        print("Loading DeiT model for attention-based cropping...")
        
        try:
            # Try loading with torch.hub
            self.model = torch.hub.load(
                'facebookresearch/deit:main',
                'deit_tiny_patch16_224',
                pretrained=True,
                force_reload=False,
                skip_validation=True
            )
        except Exception as e:
            print(f"torch.hub.load failed: {e}")
            print("Attempting to load with timm library...")
            try:
                import timm
                self.model = timm.create_model(
                    'deit_tiny_patch16_224',
                    pretrained=True
                )
            except Exception as e2:
                raise RuntimeError(f"Failed to load model with both torch.hub and timm: {e2}")
        
        # Disable fused attention for extracting attention weights
        for block in self.model.blocks:
            block.attn.fused_attn = False
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract attention from last attention layer
        nodes, _ = get_graph_node_names(self.model)
        last_attn_node = [n for n in nodes if "attn_drop" in n][-1]
        self.feature_extractor = create_feature_extractor(
            self.model, 
            return_nodes=[last_attn_node]
        )
        self.last_attn_node = last_attn_node
        
    def get_attention_map(self, img_tensor):
        """
        Extract attention map from image tensor
        
        Args:
            img_tensor: [1, 3, 224, 224] normalized image tensor
            
        Returns:
            attention_map: [14, 14] attention scores
        """
        with torch.no_grad():
            img_tensor = img_tensor.to(self.device)
            attn = self.feature_extractor(img_tensor)[self.last_attn_node].squeeze()
            
            # Get CLS token attention to patches, averaged across heads
            main_attn = attn[:, 0, 1:].mean(dim=0)  # [196]
            attention_map = main_attn.reshape(14, 14)
            
        return attention_map.cpu()
    
    def get_top_attention_regions(self, attention_map, num_regions=5):
        """
        Get top-k attention regions for cropping
        
        Args:
            attention_map: [14, 14] attention scores
            num_regions: number of regions to extract
            
        Returns:
            regions: list of (y, x) coordinates in 14x14 grid
        """
        flat_attn = attention_map.flatten()
        top_k_indices = torch.topk(flat_attn, num_regions).indices
        
        regions = []
        for idx in top_k_indices:
            y = idx // 14
            x = idx % 14
            regions.append((y.item(), x.item()))
            
        return regions
    
    def extract_attention_crops(self, img_pil, crop_ratio=0.5, num_crops=5):
        """
        Extract crops from image based on attention regions
        
        Args:
            img_pil: PIL Image
            crop_ratio: ratio of crop size to image size
            num_crops: number of crops to extract
            
        Returns:
            crops: list of PIL Image crops
        """
        # Prepare image for attention extraction
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        img_224 = img_pil.resize((224, 224), Image.BILINEAR)
        img_tensor = preprocess(img_224).unsqueeze(0)
        
        # Get attention map
        attention_map = self.get_attention_map(img_tensor)
        
        # Get top attention regions
        regions = self.get_top_attention_regions(attention_map, num_crops)
        
        # Extract crops from original image
        W, H = img_pil.size
        crop_w = int(W * crop_ratio)
        crop_h = int(H * crop_ratio)
        
        crops = []
        for y_grid, x_grid in regions:
            # Convert grid coordinates (14x14) to image coordinates
            center_y = (y_grid + 0.5) * H / 14
            center_x = (x_grid + 0.5) * W / 14
            
            # Calculate crop boundaries
            left = max(0, int(center_x - crop_w / 2))
            top = max(0, int(center_y - crop_h / 2))
            right = min(W, left + crop_w)
            bottom = min(H, top + crop_h)
            
            # Adjust if crop goes out of bounds
            if right - left < crop_w:
                left = max(0, right - crop_w)
            if bottom - top < crop_h:
                top = max(0, bottom - crop_h)
            
            crop = img_pil.crop((left, top, right, bottom))
            crops.append(crop)
        
        return crops


class AttentionCropTransform:
    """
    Transform that applies attention-based cropping similar to FiveCrop
    """
    def __init__(self, dataname, crop_ratio=0.8, num_crops=5, device='cuda'):
        self.crop_ratio = crop_ratio
        self.num_crops = num_crops
        self.dataname = dataname
        self.device = device
        
        # Don't initialize cropper here - will be lazily initialized in each worker
        self.cropper = None
        
        # Define normalization parameters
        self.mean = {
            'cifar-10': [x / 255 for x in [125.3, 123.0, 113.9]],
            'mnist': [0.1307],
            'cifar-100': [0.5071, 0.4867, 0.4408],
            'clothing1m': [0.485, 0.456, 0.406]
        }
        self.std = {
            'cifar-10': [x / 255 for x in [63.0, 62.1, 66.7]],
            'mnist': [0.3081],
            'cifar-100': [0.2675, 0.2565, 0.2761],
            'clothing1m': [0.229, 0.224, 0.225]
        }
        self.crop_size = {
            'cifar-10': 32,
            'mnist': 28,
            'cifar-100': 32,
            'clothing1m': 224
        }
        
        self.mean_ = self.mean[dataname]
        self.std_ = self.std[dataname]
        self.crop_size_ = self.crop_size[dataname]
    
    def _get_cropper(self):
        """Lazy initialization of cropper for multiprocessing compatibility"""
        if self.cropper is None:
            # In DataLoader workers, use CPU to avoid CUDA forking issues
            import multiprocessing
            if multiprocessing.current_process().name != 'MainProcess':
                device = 'cpu'
            else:
                device = self.device
            self.cropper = AttentionCropper(device=device)
        return self.cropper
        
    def __call__(self, img_pil):
        """
        Apply attention-based cropping
        
        Args:
            img_pil: PIL Image
            
        Returns:
            crops_tensor: [num_crops, C, H, W] stacked tensor of crops
        """
        # Get cropper (lazy initialization for worker safety)
        cropper = self._get_cropper()
        
        # Extract attention-based crops
        crops = cropper.extract_attention_crops(
            img_pil, 
            crop_ratio=self.crop_ratio, 
            num_crops=self.num_crops
        )
        
        # Process crops
        crop_tensors = []
        for crop in crops:
            # Resize to target size
            crop_resized = crop.resize((self.crop_size_, self.crop_size_), Image.BILINEAR)
            
            # Convert to tensor and normalize
            crop_tensor = transforms.ToTensor()(crop_resized)
            crop_tensor = transforms.Normalize(self.mean_, self.std_)(crop_tensor)
            crop_tensors.append(crop_tensor)
        
        # Stack crops
        crops_tensor = torch.stack(crop_tensors)
        
        return crops_tensor


def attention_based_crop_transform(dataname, crop_ratio=0.7, num_crops=5, device='cuda'):
    """
    Create attention-based crop transform (replacement for five_cut_transform)
    
    Args:
        dataname: dataset name
        crop_ratio: ratio of crop size to image size
        num_crops: number of crops to extract
        device: device for attention model
        
    Returns:
        transform: AttentionCropTransform instance
    """
    if dataname == 'cifar-10n':
        dataname = 'cifar-10'
    if dataname == 'cifar-100n':
        dataname = 'cifar-100'
    
    return AttentionCropTransform(
        dataname=dataname,
        crop_ratio=crop_ratio,
        num_crops=num_crops,
        device=device
    )
