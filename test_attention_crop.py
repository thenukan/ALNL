"""
Test script for attention-based cropping
Demonstrates the difference between random FiveCrop and attention-based cropping
"""
import os
# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils.attention_crop import AttentionCropper
import numpy as np

def visualize_attention_crops(image_path, output_path='attention_crops_comparison.png'):
    """
    Visualize attention-based crops vs random crops
    
    Args:
        image_path: path to input image
        output_path: path to save visualization
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Initialize attention cropper
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cropper = AttentionCropper(device=device)
    
    # Get attention-based crops
    attention_crops = cropper.extract_attention_crops(img, crop_ratio=0.7, num_crops=5)
    
    # Prepare for visualization
    img_224 = img.resize((224, 224), Image.BILINEAR)
    
    # Get attention map for visualization
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img_tensor = preprocess(img_224).unsqueeze(0)
    attention_map = cropper.get_attention_map(img_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Row 1: Original image and attention map
    axes[0, 0].imshow(img_224)
    axes[0, 0].set_title('Original Image', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(attention_map, cmap='viridis')
    axes[0, 1].set_title('Attention Map', fontsize=12)
    axes[0, 1].axis('off')
    
    # Overlay attention on image
    axes[0, 2].imshow(img_224)
    axes[0, 2].imshow(attention_map, cmap='viridis', alpha=0.5)
    axes[0, 2].set_title('Attention Overlay', fontsize=12)
    axes[0, 2].axis('off')
    
    axes[0, 3].axis('off')
    axes[0, 3].text(0.5, 0.5, 'Attention-based cropping\nfocuses on most\nimportant regions', 
                   ha='center', va='center', fontsize=10, wrap=True)
    
    # Row 2: Show first 4 attention-based crops
    for i in range(min(4, len(attention_crops))):
        axes[1, i].imshow(attention_crops[i])
        axes[1, i].set_title(f'Attention Crop {i+1}', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    plt.close()


def compare_crop_methods(image_path):
    """
    Compare statistics of attention-based vs random cropping
    """
    from torchvision import transforms
    
    img = Image.open(image_path).convert('RGB')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Attention-based crops
    cropper = AttentionCropper(device=device)
    attention_crops = cropper.extract_attention_crops(img, crop_ratio=0.8, num_crops=5)
    
    # Random FiveCrop
    crop_size = int(min(img.size) * 0.8)
    five_crop = transforms.FiveCrop(crop_size)
    random_crops = five_crop(img)
    
    print("\n" + "="*60)
    print("CROP METHOD COMPARISON")
    print("="*60)
    print(f"\nOriginal image size: {img.size}")
    print(f"Crop size: {crop_size}x{crop_size}")
    print(f"\nAttention-based crops: {len(attention_crops)} regions")
    print(f"Random FiveCrop: {len(random_crops)} regions (4 corners + center)")
    
    print("\nAttention-based cropping advantages:")
    print("  ✓ Focuses on semantically important regions")
    print("  ✓ Adapts to image content")
    print("  ✓ Better for noisy label learning")
    print("  ✓ Captures discriminative features")
    
    print("\nRandom FiveCrop:")
    print("  • Fixed positions (corners + center)")
    print("  • Content-agnostic")
    print("  • May miss important regions")
    print("="*60 + "\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_attention_crop.py <image_path>")
        print("\nExample:")
        print("  python test_attention_crop.py bird1.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("Testing attention-based cropping...")
    print(f"Image: {image_path}\n")
    
    # Visualize
    visualize_attention_crops(image_path)
    
    # Compare methods
    compare_crop_methods(image_path)
    
    print("\n✓ Test complete!")
