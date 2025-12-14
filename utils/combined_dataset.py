"""
Combined Image Dataset for Multi-Label Learning

Creates a dataset where each sample is a combination of two original images,
and the label is a multi-hot vector containing both original labels.
This creates pseudo multi-label data from single-label datasets.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from torchvision import transforms


class CombinedImageDataset(Dataset):
    """
    Dataset that creates combined images from pairs of original images.
    Each combined image gets labels from both source images.
    """
    
    def __init__(self, original_dataset, num_classes, combination_ratio=0.5, 
                 combination_method='horizontal', transform=None, max_combinations=None):
        """
        Args:
            original_dataset: Original dataset (e.g., CIFAR-10, CIFAR-100)
            num_classes: Number of classes in the dataset
            combination_ratio: Ratio of combined samples vs original samples
            combination_method: 'horizontal', 'vertical', 'blend', or 'grid'
            transform: Transform to apply after combination
            max_combinations: Maximum number of combined samples to create
        """
        self.original_dataset = original_dataset
        self.num_classes = num_classes
        self.combination_ratio = combination_ratio
        self.combination_method = combination_method
        self.transform = transform
        
        # Store original data
        self.original_data = []
        self.original_labels = []
        
        for i in range(len(original_dataset)):
            img, label = original_dataset[i]
            # Ensure PIL image format
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            elif hasattr(img, 'convert'):  # Already PIL
                pass  # Keep as is
            else:
                # Convert numpy array to PIL if needed
                img = Image.fromarray(img)
            
            self.original_data.append(img)
            self.original_labels.append(label)
        
        # Calculate how many combined samples to create
        num_original = len(self.original_data)
        num_combined = int(num_original * combination_ratio)
        if max_combinations:
            num_combined = min(num_combined, max_combinations)
        
        # Generate combined samples
        print(f"Creating {num_combined} combined samples from {num_original} original samples...")
        self.combined_data, self.combined_labels = self._create_combined_samples(num_combined)
        
        # Combine original and combined data
        self.all_data = self.original_data + self.combined_data
        self.all_labels = self._create_all_labels()
        
        print(f"Total dataset size: {len(self.all_data)} (Original: {num_original}, Combined: {num_combined})")
        
    def _create_combined_samples(self, num_combined):
        """Create combined image samples"""
        combined_data = []
        combined_labels = []
        
        for i in range(num_combined):
            # Randomly select two different images
            idx1 = random.randint(0, len(self.original_data) - 1)
            idx2 = random.randint(0, len(self.original_data) - 1)
            while idx2 == idx1:  # Ensure different images
                idx2 = random.randint(0, len(self.original_data) - 1)
            
            img1 = self.original_data[idx1]
            img2 = self.original_data[idx2]
            label1 = self.original_labels[idx1]
            label2 = self.original_labels[idx2]
            
            # Convert to PIL if needed
            if isinstance(img1, torch.Tensor):
                img1 = transforms.ToPILImage()(img1)
            if isinstance(img2, torch.Tensor):
                img2 = transforms.ToPILImage()(img2)
            
            # Combine images
            combined_img = self._combine_images(img1, img2)
            
            # Create multi-hot label
            multi_label = np.zeros(self.num_classes)
            multi_label[label1] = 1
            multi_label[label2] = 1
            
            combined_data.append(combined_img)
            combined_labels.append((label1, label2))  # Store both original labels
        
        return combined_data, combined_labels
    
    def _combine_images(self, img1, img2):
        """Combine two images based on the combination method"""
        if self.combination_method == 'horizontal':
            # Resize to half width and concatenate horizontally
            w, h = img1.size
            img1_resized = img1.resize((w//2, h))
            img2_resized = img2.resize((w//2, h))
            combined = Image.new('RGB', (w, h))
            combined.paste(img1_resized, (0, 0))
            combined.paste(img2_resized, (w//2, 0))
            
        elif self.combination_method == 'vertical':
            # Resize to half height and concatenate vertically
            w, h = img1.size
            img1_resized = img1.resize((w, h//2))
            img2_resized = img2.resize((w, h//2))
            combined = Image.new('RGB', (w, h))
            combined.paste(img1_resized, (0, 0))
            combined.paste(img2_resized, (0, h//2))
            
        elif self.combination_method == 'blend':
            # Alpha blend the two images
            combined = Image.blend(img1, img2, alpha=0.5)
            
        elif self.combination_method == 'grid':
            # 2x2 grid: img1 in top-left and bottom-right, img2 in other corners
            w, h = img1.size
            img1_resized = img1.resize((w//2, h//2))
            img2_resized = img2.resize((w//2, h//2))
            combined = Image.new('RGB', (w, h))
            combined.paste(img1_resized, (0, 0))      # Top-left
            combined.paste(img2_resized, (w//2, 0))   # Top-right
            combined.paste(img2_resized, (0, h//2))   # Bottom-left
            combined.paste(img1_resized, (w//2, h//2)) # Bottom-right
            
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return combined
    
    def _create_all_labels(self):
        """Create labels for all samples (original + combined)"""
        all_labels = []
        
        # Original samples: single-hot labels
        for label in self.original_labels:
            multi_label = np.zeros(self.num_classes, dtype=np.float32)
            multi_label[label] = 1
            all_labels.append(multi_label)
        
        # Combined samples: multi-hot labels
        for label1, label2 in self.combined_labels:
            multi_label = np.zeros(self.num_classes, dtype=np.float32)
            multi_label[label1] = 1
            multi_label[label2] = 1
            all_labels.append(multi_label)
        
        return all_labels
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        img = self.all_data[idx]
        label = self.all_labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_noise_labels(self, flip_rate=0.0, seed=42):
        """
        Generate noisy single-class labels for training compatibility.
        For combined samples, randomly choose one of the two labels.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        noise_labels = []
        
        # Original samples
        for i, label in enumerate(self.original_labels):
            if random.random() < flip_rate:
                # Flip to random label
                noise_labels.append(random.randint(0, self.num_classes - 1))
            else:
                noise_labels.append(label)
        
        # Combined samples: randomly choose one of the two labels
        for label1, label2 in self.combined_labels:
            chosen_label = random.choice([label1, label2])
            if random.random() < flip_rate:
                # Flip to random label
                noise_labels.append(random.randint(0, self.num_classes - 1))
            else:
                noise_labels.append(chosen_label)
        
        return noise_labels


def create_combined_dataset(dataset_name, data_root, combination_ratio=0.3, 
                          combination_method='horizontal', train=True):
    """
    Factory function to create combined datasets for different datasets.
    
    Args:
        dataset_name: 'cifar-10', 'cifar-100', 'mnist'
        data_root: Root directory for data
        combination_ratio: Ratio of combined vs original samples
        combination_method: Method for combining images
        train: Whether to use training or test split
    
    Returns:
        CombinedImageDataset instance
    """
    from torchvision import datasets as dsets
    from utils.utils_data import get_transform
    
    # Get the base transform (keep as PIL for combination operations)
    base_transform = None  # Keep original PIL images
    
    # Load original dataset
    if dataset_name == 'cifar-10':
        original_dataset = dsets.CIFAR10(root=data_root, train=train, 
                                       transform=base_transform, download=True)
        num_classes = 10
    elif dataset_name == 'cifar-100':
        original_dataset = dsets.CIFAR100(root=data_root, train=train, 
                                        transform=base_transform, download=True)
        num_classes = 100
    elif dataset_name == 'mnist':
        original_dataset = dsets.MNIST(root=data_root, train=train, 
                                     transform=base_transform, download=True)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Get proper transform for the dataset (but don't apply it yet)
    # final_transform = get_transform(dataname=dataset_name, train=train)
    
    # Create combined dataset without final transforms
    combined_dataset = CombinedImageDataset(
        original_dataset=original_dataset,
        num_classes=num_classes,
        combination_ratio=combination_ratio,
        combination_method=combination_method,
        transform=None  # No transform initially
    )
    
    return combined_dataset, num_classes


if __name__ == "__main__":
    # Test the combined dataset
    print("Testing combined dataset creation...")
    
    # Test with CIFAR-10
    combined_dataset, num_classes = create_combined_dataset(
        dataset_name='cifar-10',
        data_root='~/data',
        combination_ratio=0.2,
        combination_method='horizontal'
    )
    
    print(f"Dataset created with {len(combined_dataset)} samples")
    print(f"Number of classes: {num_classes}")
    
    # Get a sample
    img, multi_label = combined_dataset[0]
    print(f"Sample shape: {img.shape}")
    print(f"Multi-label: {multi_label}")
    print(f"Active classes: {np.where(multi_label == 1)[0]}")
    
    # Test noise label generation
    noise_labels = combined_dataset.get_noise_labels(flip_rate=0.1)
    print(f"Generated {len(noise_labels)} noise labels")