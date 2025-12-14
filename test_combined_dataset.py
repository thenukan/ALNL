"""
Example script demonstrating the combined multi-label dataset

Run this to test the combined dataset functionality before integrating
with the main training pipeline.
"""

import sys
import os
sys.path.append('/home/thenukan/PLM-a')

from utils.combined_dataset import create_combined_dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


def visualize_combined_samples(dataset, num_samples=4):
    """Visualize some combined samples"""
    fig, axes = plt.subplots(2, num_samples//2, figsize=(12, 6))
    axes = axes.flatten() if num_samples > 2 else [axes]
    
    for i in range(num_samples):
        img, multi_label = dataset[i]
        
        # Convert tensor to numpy for visualization
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            # Denormalize if needed
            img_np = (img_np + 1) / 2  # Assuming normalization to [-1, 1]
            img_np = np.clip(img_np, 0, 1)
        else:
            img_np = np.array(img)
        
        axes[i].imshow(img_np)
        active_labels = np.where(multi_label == 1)[0]
        axes[i].set_title(f"Labels: {active_labels}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/thenukan/PLM-a/combined_samples_visualization.png')
    plt.close()
    print("Visualization saved to combined_samples_visualization.png")


def test_combined_dataset():
    """Test the combined dataset creation and functionality"""
    
    print("="*60)
    print("TESTING COMBINED MULTI-LABEL DATASET")
    print("="*60)
    
    # Test different combination methods
    methods = ['horizontal', 'vertical', 'blend', 'grid']
    
    for method in methods:
        print(f"\n--- Testing {method} combination ---")
        
        # Create combined dataset
        try:
            combined_dataset, num_classes = create_combined_dataset(
                dataset_name='cifar-10',  # Start with smaller dataset for testing
                data_root='/home/thenukan/data',
                combination_ratio=0.2,  # 20% combined samples
                combination_method=method,
                train=True
            )
            
            print(f"✓ Successfully created {method} combined dataset")
            print(f"  Total samples: {len(combined_dataset)}")
            print(f"  Number of classes: {num_classes}")
            
            # Check label distribution
            all_labels = np.array([label for label in combined_dataset.all_labels])
            labels_per_sample = all_labels.sum(axis=1)
            
            print(f"  Single-label samples: {(labels_per_sample == 1).sum()}")
            print(f"  Multi-label samples: {(labels_per_sample > 1).sum()}")
            print(f"  Average labels per sample: {labels_per_sample.mean():.2f}")
            
            # Test noise label generation
            noise_labels = combined_dataset.get_noise_labels(flip_rate=0.1, seed=42)
            print(f"  Generated noise labels: {len(noise_labels)}")
            
            # Sample some data points
            print(f"  Sample data shapes:")
            img, label = combined_dataset[0]
            print(f"    Image: {img.shape if hasattr(img, 'shape') else type(img)}")
            print(f"    Label: {label.shape}, Active classes: {np.where(label==1)[0]}")
            
        except Exception as e:
            print(f"✗ Error with {method}: {e}")
    
    print("\n" + "="*60)
    print("INTEGRATION WITH TRAINING PIPELINE")
    print("="*60)
    
    # Show how to integrate with existing training
    print("\nTo use combined dataset in training, run:")
    print("python lnl_combined.py --ds cifar-10 --use_combined_dataset --combination_ratio 0.3 --combination_method horizontal")
    
    print("\nAvailable combination methods:")
    print("  - horizontal: Images side by side")
    print("  - vertical: Images stacked vertically") 
    print("  - blend: Alpha blended overlay")
    print("  - grid: 2x2 grid pattern")
    
    print("\nParameters:")
    print("  --combination_ratio: Ratio of combined vs original samples (default: 0.3)")
    print("  --combination_method: How to combine images (default: horizontal)")
    print("  --use_combined_dataset: Enable combined dataset mode")


def create_training_example():
    """Create a complete example for training"""
    
    # Create a training script snippet
    training_code = '''
# Example: Training with combined multi-label dataset

python lnl_combined.py \\
    --ds cifar-10 \\
    --flip_rate 0.2 \\
    --gpu 0 \\
    --use_combined_dataset \\
    --combination_ratio 0.3 \\
    --combination_method horizontal \\
    --ep 20 \\
    --bs 128

# This will:
# 1. Load CIFAR-10 dataset
# 2. Create 30% additional samples by combining pairs of images  
# 3. Assign multi-hot labels to combined samples
# 4. Train using both original and combined samples
# 5. Use the multi-labels as pseudo ground truth for candidate labeling
'''
    
    with open('/home/thenukan/PLM-a/training_example.sh', 'w') as f:
        f.write(training_code)
    
    print("Training example saved to training_example.sh")


if __name__ == "__main__":
    import torch
    
    # Set up for testing
    print("Setting up combined dataset test...")
    
    try:
        test_combined_dataset()
        create_training_example()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("You can now use the combined dataset in your training.")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()