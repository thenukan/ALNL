"""
Iterative Clean-Sample Refinement Approach for Learning with Noisy Labels

This module implements a novel training pipeline that progressively refines 
clean samples through K-means clustering and iterative model training, 
instead of directly learning from noisy labeled data.

Pipeline:
1. Initial filtering using K-means clustering on features to identify outliers
2. Train ResNet on filtered clean samples
3. Iteratively refine: use learned representations to re-cluster and update clean set
4. Continue until label consistency stabilizes
"""

import os
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn.functional as F
import torch.utils.data

from models.models import get_model
from utils.utils_algo import accuracy_check, one_hot, init_gpuseed, accuracy_check_noise
from utils.utils_algo import get_scheduler
from utils.utils_data import get_origin_datasets, indices_split, get_transform, get_noise_dataset


def filter_by_class_distance(features, labels, keep_ratio, num_classes):
    """
    Filter samples per class based on distance from class mean.
    Keep only the closest samples (outliers ignored).
    
    Args:
        features: Feature representations (N, D)
        labels: Noisy labels (N,)
        keep_ratio: Ratio of samples to keep per class (e.g., 0.5 = keep 50%)
        num_classes: Number of classes
        
    Returns:
        clean_mask: Boolean mask indicating samples to keep
    """
    clean_mask = np.zeros(len(labels), dtype=bool)
    
    for class_id in range(num_classes):
        class_indices = np.where(labels == class_id)[0]
        
        if len(class_indices) == 0:
            continue
            
        class_features = features[class_indices]
        
        # Compute distance from class mean
        mean_feature = class_features.mean(axis=0)
        distances = np.linalg.norm(class_features - mean_feature, axis=1)
        
        # Keep only the closest samples based on keep_ratio
        n_keep = max(1, int(len(class_indices) * keep_ratio))
        sorted_indices = np.argsort(distances)[:n_keep]
        
        clean_mask[class_indices[sorted_indices]] = True
    
    kept_samples = clean_mask.sum()
    total_samples = len(clean_mask)
    print(f"  Per-class filtering: Keeping {kept_samples}/{total_samples} samples "
          f"({100*kept_samples/total_samples:.1f}%) - Target: {keep_ratio*100:.0f}%")
    
    return clean_mask


class CleanSampleRefiner:
    """
    Manages the iterative process of identifying clean samples through clustering
    and model-based feature refinement.
    """
    
    def __init__(self, num_classes, device, confidence_threshold=0.8, outlier_percentile=10):
        """
        Args:
            num_classes: Number of classes in the dataset
            device: torch device for computations
            confidence_threshold: Minimum confidence score to consider a sample clean
            outlier_percentile: Percentile threshold for outlier detection (lower = more aggressive)
        """
        self.num_classes = num_classes
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.outlier_percentile = outlier_percentile
        self.clean_mask = None
        self.sample_weights = None
        
    def extract_features(self, model, dataloader):
        """
        Extract deep features from the model's penultimate layer.
        
        Args:
            model: Trained neural network model
            dataloader: DataLoader for the dataset
            
        Returns:
            features: numpy array of shape (N, feature_dim)
            labels: numpy array of shape (N,)
            indices: numpy array of sample indices
        """
        model.eval()
        features_list = []
        labels_list = []
        indices_list = []
        
        # Hook to capture features before final FC layer
        features_hook = []
        def hook_fn(module, input, output):
            features_hook.append(input[0].detach())
        
        # Register hook on the final FC layer
        if hasattr(model, 'fc'):
            handle = model.fc.register_forward_hook(hook_fn)
        elif hasattr(model, 'linear'):
            handle = model.linear.register_forward_hook(hook_fn)
        else:
            raise ValueError("Model structure not recognized for feature extraction")
        
        with torch.no_grad():
            for images, labels, inds in dataloader:
                images = images.to(self.device, non_blocking=True)
                _ = model(images)
                
                # Get features from hook
                features = features_hook[-1]
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
                indices_list.append(inds.numpy())
                
        handle.remove()
        
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        indices = np.concatenate(indices_list)
        
        return features, labels, indices
    
    def cluster_based_filtering(self, features, labels, iteration=0):
        """
        Perform K-means clustering within each class to identify outliers.
        
        Args:
            features: Feature representations (N, D)
            labels: Noisy labels (N,)
            iteration: Current iteration number (affects aggressiveness)
            
        Returns:
            clean_mask: Boolean mask indicating clean samples
            cluster_confidence: Confidence scores for each sample
        """
        clean_mask = np.zeros(len(labels), dtype=bool)
        cluster_confidence = np.zeros(len(labels))
        
        # Adjust outlier threshold based on iteration (start conservative, get more aggressive)
        adjusted_percentile = min(self.outlier_percentile + iteration * 5, 30)
        
        for class_id in range(self.num_classes):
            class_indices = np.where(labels == class_id)[0]
            
            if len(class_indices) < 10:  # Too few samples to cluster
                clean_mask[class_indices] = True
                cluster_confidence[class_indices] = 1.0
                continue
            
            class_features = features[class_indices]
            
            # Determine optimal number of clusters (1 to 3 clusters per class)
            n_clusters = min(max(1, len(class_indices) // 20), 3)
            
            if n_clusters == 1:
                # Single cluster - use distance from mean to detect outliers
                mean_feature = class_features.mean(axis=0)
                distances = np.linalg.norm(class_features - mean_feature, axis=1)
                threshold = np.percentile(distances, 100 - adjusted_percentile)
                
                clean_indices = class_indices[distances <= threshold]
                clean_mask[clean_indices] = True
                
                # Confidence based on inverse distance
                max_dist = distances.max() + 1e-8
                cluster_confidence[class_indices] = 1.0 - (distances / max_dist)
                
            else:
                # Multiple clusters - use clustering quality
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(class_features)
                
                # Compute distances to cluster centers
                distances = np.zeros(len(class_features))
                for i in range(n_clusters):
                    cluster_mask = cluster_labels == i
                    cluster_center = kmeans.cluster_centers_[i]
                    distances[cluster_mask] = np.linalg.norm(
                        class_features[cluster_mask] - cluster_center, axis=1
                    )
                
                # Identify clean samples based on distance threshold
                threshold = np.percentile(distances, 100 - adjusted_percentile)
                clean_local_mask = distances <= threshold
                
                clean_mask[class_indices[clean_local_mask]] = True
                
                # Confidence based on inverse distance and cluster compactness
                max_dist = distances.max() + 1e-8
                cluster_confidence[class_indices] = 1.0 - (distances / max_dist)
        
        print(f"  Clustering: {clean_mask.sum()}/{len(clean_mask)} samples marked as clean "
              f"({100*clean_mask.mean():.1f}%)")
        
        return clean_mask, cluster_confidence
    
    def model_based_filtering(self, model, dataloader, labels):
        """
        Use model predictions to identify confident samples.
        
        Args:
            model: Trained model
            dataloader: DataLoader
            labels: Noisy labels
            
        Returns:
            confidence_mask: Boolean mask for confident predictions
            prediction_probs: Prediction probabilities for each sample
        """
        model.eval()
        all_probs = []
        all_indices = []
        
        with torch.no_grad():
            for images, _, inds in dataloader:
                images = images.to(self.device, non_blocking=True)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                all_indices.append(inds)
        
        all_probs = torch.cat(all_probs, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        # Check if predicted label matches noisy label with high confidence
        predicted_labels = all_probs.argmax(dim=1).numpy()
        max_probs = all_probs.max(dim=1)[0].numpy()
        
        # Sort back to original order
        sorted_indices = torch.argsort(all_indices)
        predicted_labels = predicted_labels[sorted_indices]
        max_probs = max_probs[sorted_indices]
        
        # Sample is confident if: high confidence AND matches noisy label
        confidence_mask = (max_probs >= self.confidence_threshold) & (predicted_labels == labels)
        
        print(f"  Model filtering: {confidence_mask.sum()}/{len(confidence_mask)} samples are confident "
              f"({100*confidence_mask.mean():.1f}%)")
        
        return confidence_mask, all_probs[sorted_indices].numpy()
    
    def combine_filters(self, cluster_mask, model_mask, cluster_conf, model_conf):
        """
        Combine clustering and model-based filtering results.
        
        Args:
            cluster_mask: Clean mask from clustering
            model_mask: Confidence mask from model
            cluster_conf: Clustering confidence scores
            model_conf: Model prediction confidences
            
        Returns:
            final_mask: Combined clean sample mask
            final_weights: Sample weights for training
        """
        # Use union for early iterations, intersection for later ones
        # This allows progressive refinement
        final_mask = cluster_mask | model_mask  # Union: accept if either method says clean
        
        # Combine confidence scores
        final_weights = (cluster_conf + model_conf.max(axis=1)) / 2.0
        final_weights = final_weights * final_mask  # Zero out filtered samples
        
        # Normalize weights
        if final_weights.sum() > 0:
            final_weights = final_weights / final_weights.sum() * len(final_weights)
        
        print(f"  Combined filtering: {final_mask.sum()}/{len(final_mask)} samples selected "
              f"({100*final_mask.mean():.1f}%)")
        
        return final_mask, final_weights


def prenp(args, paths, noise_labels):
    """
    Progressive Noisy-label Purification through Iterative Refinement.
    
    Instead of directly training on noisy labels, this function:
    1. Filters outliers using K-means clustering
    2. Trains model on clean subset
    3. Iteratively refines clean set using learned representations
    4. Generates candidate labels for downstream tasks
    
    Args:
        args: Training arguments
        paths: Dictionary of file paths for saving/loading
        noise_labels: Array of noisy labels
    """
    if os.path.exists(paths['multi_labels']) and os.path.exists(paths['pre_model']):
        print("Pre-trained model and labels exist. Loading...")
        return

    # seed and device
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)
    
    # Fix for Windows multiprocessing
    import sys
    if sys.platform == 'win32':
        args.num_workers = 0
        use_persistent_workers = False
    else:
        use_persistent_workers = True if args.num_workers > 0 else False

    # transform
    train_transform = get_transform(dataname=args.ds, train=True)
    val_transform = get_transform(dataname=args.ds, train=False)

    # dataset
    ordinary_train_dataset, test_dataset, num_classes = get_origin_datasets(
        dataname=args.ds, transform=val_transform, data_root=args.data_root
    )

    train_indices, val_indices = indices_split(
        len_dataset=len(ordinary_train_dataset), seed=args.seed, val_ratio=0.1
    )

    # Create refiner for feature extraction
    refiner = CleanSampleRefiner(
        num_classes=num_classes, 
        device=device,
        confidence_threshold=0.7,
        outlier_percentile=15
    )

    # Create datasets
    val_dataset = get_noise_dataset(
        dataset=ordinary_train_dataset,
        noise_labels=noise_labels,
        transformations=val_transform,
        indices=val_indices
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, 
        persistent_workers=use_persistent_workers
    )

    # Initialize model
    model = get_model(model_name=args.mo, num_classes=num_classes, fix_backbone=False)
    model = model.to(device)
    best_model = deepcopy(model)
    best_test_acc = 0.0
    
    # Progressive filtering strategy
    # Phase 1: Keep 50%, train 15 epochs
    # Phase 2: Keep 80%, train 15 epochs
    # Phase 3+: Keep 90%+, train 15 epochs each
    
    filtering_schedule = [
        {'keep_ratio': 0.50, 'epochs': 50, 'name': 'Initial Filtering (50%)'},
        {'keep_ratio': 0.80, 'epochs': 50, 'name': 'Second Phase (80%)'},
        {'keep_ratio': 0.90, 'epochs': 50, 'name': 'Third Phase (90%)'},
        {'keep_ratio': 0.95, 'epochs': 55, 'name': 'Final Refinement (95%)'},
    ]
    
    clean_mask = np.ones(len(train_indices), dtype=bool)  # Start with all samples
    
    print("\n" + "="*80)
    print("PROGRESSIVE SAMPLE FILTERING & TRAINING")
    print("="*80)
    
    for iteration, schedule in enumerate(filtering_schedule):
        print(f"\n{'='*80}")
        print(f"PHASE {iteration + 1}/{len(filtering_schedule)}: {schedule['name']}")
        print(f"Target: Keep {schedule['keep_ratio']*100:.0f}% of samples per class")
        print(f"{'='*80}")
        
        # Create dataset with current clean mask
        current_train_indices = np.array(train_indices)[clean_mask].tolist()
        
        train_dataset = get_noise_dataset(
            dataset=ordinary_train_dataset,
            noise_labels=noise_labels,
            transformations=train_transform,
            indices=current_train_indices
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.bs, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=use_persistent_workers
        )
        
        # Also need a loader without augmentation for feature extraction
        train_eval_dataset = get_noise_dataset(
            dataset=ordinary_train_dataset,
            noise_labels=noise_labels,
            transformations=val_transform,
            indices=train_indices
        )
        
        train_eval_loader = torch.utils.data.DataLoader(
            dataset=train_eval_dataset, batch_size=args.bs, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=use_persistent_workers
        )
        
        print(f"\nTraining on {len(current_train_indices)} clean samples...")
        
        # Re-initialize model for each iteration (or use previous as warm start)
        if iteration > 0:
            # Warm start from previous iteration
            model.load_state_dict(best_model.state_dict())
        
        # Optimizer and scheduler
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, 
            weight_decay=args.wd, momentum=args.momentum,
            nesterov=args.nesterov
        )
        
        # Use epochs from schedule
        epochs_per_iteration = schedule['epochs']
        scheduler = get_scheduler(args.ds, optimizer, epochs_per_iteration)
        val_accuracy_best = 0.0
        
        for epoch in range(epochs_per_iteration):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for images, labels, inds in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, dtype=torch.int64, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(labels)
            
            scheduler.step()
            train_loss = train_loss / len(train_dataset)
            
            # Validation
            model.eval()
            val_accuracy = accuracy_check_noise(loader=val_loader, model=model, device=device)
            test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
            
            if epoch % 5 == 0 or epoch == epochs_per_iteration - 1:
                print(f'  Epoch {epoch}/{epochs_per_iteration}: '
                      f'Loss={train_loss:.4f}, Val={val_accuracy:.4f}, Test={test_accuracy:.4f}')
            
            if val_accuracy > val_accuracy_best:
                val_accuracy_best = val_accuracy
                best_model = deepcopy(model)
                
            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
        
        print(f"\nPhase {iteration + 1} completed. Best Val Acc: {val_accuracy_best:.4f}, "
              f"Best Test Acc: {best_test_acc:.4f}")
        
        # Update clean mask for next iteration (except last iteration)
        if iteration < len(filtering_schedule) - 1:
            print(f"\nFiltering samples for next phase...")
            
            # Extract features using best model from this phase
            features, labels_array, _ = refiner.extract_features(best_model, train_eval_loader)
            
            # Apply per-class filtering based on target keep_ratio
            next_keep_ratio = filtering_schedule[iteration + 1]['keep_ratio']
            clean_mask = filter_by_class_distance(
                features, noise_labels[train_indices], next_keep_ratio, num_classes
            )
    
    print(f"\n{'='*80}")
    print("FINAL STAGE: Generating Candidate Labels")
    print(f"{'='*80}\n")
    
    # Final evaluation and label generation
    best_model.eval()
    
    # Generate multi-labels using augmentation voting
    from utils.utils_data import five_cut_transform
    use_attention = getattr(args, 'use_attention_crop', False)
    if use_attention:
        print("Using attention-based cropping for candidate label generation")
    crop_transform = five_cut_transform(args.ds, crop_ratio=args.crop_ratio,
                                       use_attention=use_attention, device=device)
    
    es_crop_dataset = get_noise_dataset(
        dataset=ordinary_train_dataset,
        noise_labels=noise_labels,
        transformations=crop_transform
    )
    
    estimate_crop_dataloader = torch.utils.data.DataLoader(
        dataset=es_crop_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    
    es_dataset = get_noise_dataset(
        dataset=ordinary_train_dataset,
        noise_labels=noise_labels,
        transformations=val_transform,
        indices=train_indices
    )
    
    estimate_dataloader = torch.utils.data.DataLoader(
        dataset=es_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    
    # Generate candidate labels with augmentation
    labels_multi = np.zeros([len(ordinary_train_dataset), num_classes])
    outputs_all = torch.zeros(len(es_dataset), num_classes).to(device)
    
    with torch.no_grad():
        for images, labels, inds in estimate_crop_dataloader:
            bs, ncrops, c, h, w = images.size()
            images = images.to(device, non_blocking=True)
            probs = best_model(images.view(-1, c, h, w))
            probs_ncrops = probs.view(bs, ncrops, -1)
            
            # Vote across crops
            labels_temp = (probs_ncrops == probs_ncrops.max(dim=2, keepdim=True)[0]).to(dtype=torch.int32)
            labels_temp = labels_temp.sum(1)
            labels_multi[inds, :] = labels_temp.cpu().numpy()
    
    # Get final probabilities
    with torch.no_grad():
        for images, labels, inds in estimate_dataloader:
            images = images.to(device, non_blocking=True)
            outputs = best_model(images)
            outputs_all[inds, :] = outputs
    
    probs_all = F.softmax(outputs_all, dim=1)
    
    # Binarize multi-labels
    labels_multi[labels_multi > 1] = 1
    
    # Save results
    np.save(paths['pre_probs'], probs_all.cpu().numpy())
    np.save(paths['multi_labels'], labels_multi)
    torch.save(best_model.state_dict(), paths['pre_model'])
    
    # Add noisy label to candidate set
    noise_oh = one_hot(noise_labels)
    labels_multi = labels_multi + noise_oh
    labels_multi[labels_multi > 1] = 1
    
    # Calculate hit ratio
    hit_ratio = np.array([
        labels_multi[i][ordinary_train_dataset.targets[i]] 
        for i in range(len(labels_multi))
    ]).mean()
    
    mean_labels = labels_multi.mean() * labels_multi.shape[1]
    
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Final Test Accuracy: {best_test_acc:.4f}")
    print(f"Hit Ratio: {hit_ratio:.4f}")
    print(f"Mean Candidate Labels per Sample: {mean_labels:.2f}")
    print(f"Clean Sample Ratio: {clean_mask.mean():.2%}")
    print(f"{'='*80}\n")
