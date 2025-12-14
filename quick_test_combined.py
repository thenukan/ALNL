"""
Quick test to verify combined dataset integration
"""

import os
import sys
sys.path.append('/home/thenukan/PLM-a')

import numpy as np
import torch

# Test minimal combined dataset functionality
def test_combined_integration():
    print("Testing combined dataset integration...")
    
    try:
        from utils.combined_dataset import create_combined_dataset
        
        # Create a small combined dataset for testing
        combined_dataset, num_classes = create_combined_dataset(
            dataset_name='cifar-10',
            data_root='/home/thenukan/data',
            combination_ratio=0.1,  # Small ratio for testing
            combination_method='horizontal',
            train=True
        )
        
        print(f"✓ Combined dataset created: {len(combined_dataset)} samples, {num_classes} classes")
        
        # Test the wrapper class
        class CombinedDatasetWrapper:
            """Wrapper to make CombinedImageDataset compatible with get_cantar_dataset"""
            def __init__(self, combined_dataset):
                self.combined_dataset = combined_dataset
                self.targets = combined_dataset.get_noise_labels(flip_rate=0, seed=42)
                # Create data attribute for compatibility
                self.data = []
                for i in range(len(combined_dataset)):
                    img, _ = combined_dataset[i]
                    self.data.append(img)
                self.data = np.array(self.data)
            
            def __len__(self):
                return len(self.combined_dataset)
            
            def __getitem__(self, idx):
                return self.combined_dataset[idx]
        
        # Test wrapper
        wrapped_dataset = CombinedDatasetWrapper(combined_dataset)
        
        print(f"✓ Wrapper created: {len(wrapped_dataset)} samples")
        print(f"✓ Data shape: {len(wrapped_dataset.data)} items")
        print(f"✓ Targets shape: {len(wrapped_dataset.targets)} labels")
        
        # Test accessing multi-labels
        multi_labels = np.array([label for label in wrapped_dataset.combined_dataset.all_labels])
        print(f"✓ Multi-labels shape: {multi_labels.shape}")
        print(f"✓ Average labels per sample: {multi_labels.sum(axis=1).mean():.2f}")
        
        # Test noise label generation
        noise_labels = wrapped_dataset.combined_dataset.get_noise_labels(flip_rate=0.1, seed=42)
        print(f"✓ Noise labels generated: {len(noise_labels)}")
        
        print("\n✅ All tests passed! Combined dataset integration should work.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_combined_integration()
    if success:
        print("\nYou can now run:")
        print("python lnl.py --ds cifar-10 --use_combined_dataset --combination_ratio 0.3 --num_workers 4")
    else:
        print("\nPlease check the combined dataset implementation.")