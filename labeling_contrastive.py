import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from models.models import get_model
from utils.utils_algo import accuracy_check, one_hot, init_gpuseed, \
    accuracy_check_noise
from utils.utils_algo import get_scheduler
from utils.utils_data import get_origin_datasets, five_cut_transform, indices_split, \
    get_transform, get_noise_dataset, BalancedSampler


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to encourage intra-class similarity and inter-class separation
    """
    def __init__(self, temperature=0.5, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim] - normalized feature vectors
        labels: [batch_size] - predicted labels (potentially noisy)
        """
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for same class pairs
        labels = labels.unsqueeze(1)
        mask_same_class = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask_same_class = mask_same_class - torch.eye(labels.size(0), device=labels.device)
        
        # Compute contrastive loss
        # Positive pairs: same class (should be similar)
        pos_similarity = similarity_matrix * mask_same_class
        
        # Negative pairs: different class (should be dissimilar)
        mask_diff_class = 1 - mask_same_class - torch.eye(labels.size(0), device=labels.device)
        neg_similarity = similarity_matrix * mask_diff_class
        
        # Loss: maximize intra-class similarity, minimize inter-class similarity
        pos_loss = -torch.log(torch.exp(pos_similarity).sum(1) + 1e-8)
        neg_loss = torch.log(torch.exp(neg_similarity).sum(1) + 1e-8)
        
        loss = (pos_loss + neg_loss).mean()
        
        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    More robust to label noise by focusing on feature consistency
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Remove diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        
        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class IntraClassDistanceFilter:
    """
    Filter out samples with high intra-class distance (likely mislabeled)
    """
    def __init__(self, num_classes, percentile=80):
        self.num_classes = num_classes
        self.percentile = percentile
        self.class_centers = None
    
    def update_centers(self, features, labels):
        """
        Update class centers based on feature embeddings
        """
        self.class_centers = []
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                center = features[mask].mean(dim=0)
                self.class_centers.append(center)
            else:
                self.class_centers.append(torch.zeros_like(features[0]))
        self.class_centers = torch.stack(self.class_centers)
    
    def get_clean_mask(self, features, labels):
        """
        Return mask of likely clean samples based on distance to class center
        """
        if self.class_centers is None:
            return torch.ones(len(labels), dtype=torch.bool, device=labels.device)
        
        distances = []
        for i in range(len(features)):
            label = labels[i]
            center = self.class_centers[label]
            dist = F.cosine_similarity(features[i].unsqueeze(0), center.unsqueeze(0))
            distances.append(dist.item())
        
        distances = np.array(distances)
        threshold = np.percentile(distances, 100 - self.percentile)
        
        clean_mask = torch.tensor(distances >= threshold, device=labels.device)
        return clean_mask


def prenp_contrastive(args, paths, noise_labels):
    """
    Pre-training with contrastive learning to handle label noise
    """
    if os.path.exists(paths['multi_labels']) and os.path.exists(paths['pre_model']):
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
    crop_transform = five_cut_transform(args.ds, crop_ratio=args.crop_ratio)

    # dataset
    ordinary_train_dataset, test_dataset, num_classes = get_origin_datasets(
        dataname=args.ds, transform=val_transform, data_root=args.data_root
    )

    train_indices, val_indices = indices_split(
        len_dataset=len(ordinary_train_dataset),
        seed=args.seed,
        val_ratio=0.1
    )

    # noise dataset & dataloader
    train_dataset = get_noise_dataset(
        dataset=ordinary_train_dataset,
        noise_labels=noise_labels,
        transformations=train_transform,
        indices=train_indices
    )
    es_crop_dataset = get_noise_dataset(
        dataset=ordinary_train_dataset,
        noise_labels=noise_labels,
        transformations=crop_transform
    )
    es_dataset = get_noise_dataset(
        dataset=ordinary_train_dataset,
        noise_labels=noise_labels,
        transformations=val_transform,
        indices=train_indices
    )
    val_dataset = get_noise_dataset(
        dataset=ordinary_train_dataset,
        noise_labels=noise_labels,
        transformations=val_transform,
        indices=val_indices
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    estimate_dataloader = torch.utils.data.DataLoader(
        dataset=es_dataset, batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    estimate_crop_dataloader = torch.utils.data.DataLoader(
        dataset=es_crop_dataset, batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=use_persistent_workers
    )

    # labeling
    labels_multi = np.zeros([len(ordinary_train_dataset), num_classes])
    outputs_all = torch.zeros(len(es_dataset), num_classes).to(device)

    # model with projection head for contrastive learning
    model = get_model(model_name=args.mo, num_classes=num_classes, fix_backbone=False)
    
    # Add projection head for contrastive learning
    if hasattr(model, 'fc'):
        feature_dim = model.fc.in_features
    elif hasattr(model, 'linear'):
        feature_dim = model.linear.in_features
    else:
        feature_dim = 512  # default
    
    projection_head = nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(),
        nn.Linear(feature_dim, 128)  # projection to 128-dim space
    ).to(device)
    
    model = model.to(device)
    best_model = deepcopy(model)

    # Loss functions
    ce_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = SupConLoss(temperature=0.07)
    
    # Intra-class distance filter
    distance_filter = IntraClassDistanceFilter(num_classes, percentile=80)
    
    # Hyperparameters for loss weighting
    lambda_ce = getattr(args, 'lambda_ce', 0.5)  # Weight for CE loss
    lambda_con = getattr(args, 'lambda_con', 0.5)  # Weight for contrastive loss
    warmup_epochs = getattr(args, 'warmup_epochs', 1)  # Epochs before filtering

    # opt
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(projection_head.parameters()),
        lr=args.lr, weight_decay=args.wd, momentum=args.momentum,
        nesterov=args.nesterov
    )
    scheduler = get_scheduler(args.ds, optimizer, args.ep)

    # train
    val_accuracy_best = 0
    if args.ds == 'clothing1m':
        b_sampler = BalancedSampler(num_classes, np.array(noise_labels[train_indices]))
    
    for epoch in range(args.ep):
        if args.ds == 'clothing1m':
            sampled_indices = b_sampler.ind_unsampling()
            term_indices = np.array(train_indices)[sampled_indices].tolist()
            train_dataset = get_noise_dataset(
                dataset=ordinary_train_dataset,
                noise_labels=noise_labels,
                transformations=train_transform,
                indices=term_indices
            )
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=args.bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=use_persistent_workers
            )
        
        model.train()
        projection_head.train()
        
        # Collect features for distance filtering (after warmup)
        if epoch == warmup_epochs:
            all_features = []
            all_labels = []
            model.eval()
            with torch.no_grad():
                for images, labels, inds in estimate_dataloader:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, dtype=torch.int64, non_blocking=True)
                    
                    # Extract features before final layer (same as in training loop)
                    if hasattr(model, 'fc'):
                        features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(F.relu(model.bn1(model.conv1(images))))))))
                        features = torch.flatten(features, 1)
                    else:
                        features = model(images)
                    
                    all_features.append(features)
                    all_labels.append(labels)
            
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            distance_filter.update_centers(all_features, all_labels)
            model.train()
        
        epoch_ce_loss = 0
        epoch_con_loss = 0
        
        for images, labels, inds in train_loader:
            X = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.int64, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X)
            
            # Extract features for contrastive loss
            if hasattr(model, 'fc'):
                # For ResNet-style models
                features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(F.relu(model.bn1(model.conv1(X))))))))
                features = torch.flatten(features, 1)
            else:
                # Use penultimate layer features
                features = outputs
            
            # Project features for contrastive learning
            projected_features = projection_head(features)
            
            # Apply distance-based filtering after warmup
            if epoch >= warmup_epochs and distance_filter.class_centers is not None:
                clean_mask = distance_filter.get_clean_mask(features.detach(), labels)
                
                if clean_mask.sum() > 1:  # Need at least 2 samples for contrastive
                    labels_clean = labels[clean_mask]
                    outputs_clean = outputs[clean_mask]
                    projected_features_clean = projected_features[clean_mask]
                    
                    # Cross-entropy loss only on clean samples
                    loss_ce = ce_criterion(outputs_clean, labels_clean)
                    
                    # Contrastive loss only on clean samples
                    loss_con = contrastive_criterion(projected_features_clean, labels_clean)
                else:
                    # Fallback if filtering is too aggressive
                    loss_ce = ce_criterion(outputs, labels)
                    loss_con = contrastive_criterion(projected_features, labels)
            else:
                # Before warmup, use all samples
                loss_ce = ce_criterion(outputs, labels)
                loss_con = contrastive_criterion(projected_features, labels)
            
            # Combined loss
            loss = lambda_ce * loss_ce + lambda_con * loss_con
            
            loss.backward()
            optimizer.step()
            
            epoch_ce_loss += loss_ce.item()
            epoch_con_loss += loss_con.item()

        scheduler.step()
        model.eval()

        val_accuracy = accuracy_check_noise(loader=val_dataloader, model=model, device=device)
        test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
        
        print('Epoch: {}. Val_Acc: {:.4f}. Test_Acc: {:.4f}. CE_Loss: {:.4f}. Con_Loss: {:.4f}'.format(
            epoch, val_accuracy, test_accuracy, 
            epoch_ce_loss / len(train_loader), 
            epoch_con_loss / len(train_loader)
        ))

        if val_accuracy_best < val_accuracy:
            val_accuracy_best = val_accuracy
            best_model = deepcopy(model)

    # Generate multi-labels and probabilities using best model
    best_model.eval()
    with torch.no_grad():
        for images, labels, inds in estimate_crop_dataloader:
            bs, ncrops, c, h, w = images.size()
            images = images.to(device, non_blocking=True)
            probs = best_model(images.view(-1, c, h, w))
            probs_ncrops = probs.view(bs, ncrops, -1)

            labels_temp = (probs_ncrops == probs_ncrops.max(dim=2, keepdim=True)[0]).to(dtype=torch.int32)
            labels_temp = labels_temp.sum(1)
            labels_multi[inds, :] = labels_temp.cpu().numpy()

    with torch.no_grad():
        for images, labels, inds in estimate_dataloader:
            images = images.to(device, non_blocking=True)
            outputs = best_model(images)
            outputs_all[inds, :] = outputs
    probs_all = F.softmax(outputs_all, dim=1)

    labels_multi[labels_multi > 1] = 1
    np.save(paths['pre_probs'], probs_all.cpu().numpy())
    np.save(paths['multi_labels'], labels_multi)
    torch.save(best_model.state_dict(), paths['pre_model'])

    noise_oh = one_hot(noise_labels)
    labels_multi = labels_multi + noise_oh
    labels_multi[labels_multi > 1] = 1
    print("Hit Ratio:",
          (np.array([labels_multi[i][ordinary_train_dataset.targets[i]] for i in range(len(labels_multi))]).mean()))
    print("Mean Labels:", labels_multi.mean() * labels_multi.shape[1])
