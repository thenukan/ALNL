import argparse
import os
import time
from copy import deepcopy

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch.nn.functional as F
import torch.utils.data

#from labeling import prenp
from labeling_refined_km import prenp  # Use refined approach
#from labeling_contrastive import prenp_contrastive as prenp
from models.models import get_model
from utils.utils_algo import accuracy_check, get_paths, init_gpuseed
from utils.utils_algo import get_scheduler
from utils.utils_data import get_origin_datasets, indices_split, generate_noise_labels, get_transform, \
    get_cantar_dataset, BalancedSampler

# Import combined dataset functionality
try:
    from utils.combined_dataset import create_combined_dataset
    COMBINED_DATASET_AVAILABLE = True
except ImportError:
    COMBINED_DATASET_AVAILABLE = False
    print("Combined dataset functionality not available. Install required dependencies.")


class CombinedDatasetWrapper:
    """Wrapper to make CombinedImageDataset compatible with get_cantar_dataset"""
    def __init__(self, combined_dataset, base_transform=None):
        self.combined_dataset = combined_dataset
        self.base_transform = base_transform
        self.targets = np.array(combined_dataset.get_noise_labels(flip_rate=0, seed=42))  # Original labels
        
        # Create data attribute for compatibility - store PIL images
        raw_data = []
        for i in range(len(combined_dataset)):
            # Get raw image from combined dataset
            img = combined_dataset.all_data[i]
            # Ensure it's PIL Image (should already be from our updates)
            if hasattr(img, 'numpy'):  # If it's a tensor
                from torchvision import transforms
                img = transforms.ToPILImage()(img)
            raw_data.append(img)
        
        # Store as numpy array of objects for proper indexing
        self.data = np.array(raw_data, dtype=object)
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        # Handle both single index and list of indices
        if isinstance(idx, (list, np.ndarray)):
            return [self.__getitem__(i) for i in idx]
        
        img = self.data[idx]
        label = self.combined_dataset.all_labels[idx]
        
        if self.base_transform:
            img = self.base_transform(img)
            
        return img, label


# Remove the separate IndexableDataArray class as it's not needed


class CombinedCanTarDataset:
    """Custom dataset for combined multi-label data"""
    def __init__(self, combined_dataset, candidate_labels, targets, indices, transformations, return_index=False):
        self.combined_dataset = combined_dataset
        self.indices = np.array(indices)  # Ensure indices are numpy array
        self.transformations = transformations
        self.return_index = return_index
        
        # Extract data for the given indices
        self.data = [combined_dataset.all_data[i] for i in indices]
        self.candidate_labels = np.array(candidate_labels)[self.indices]  # Ensure numpy indexing
        self.targets = np.array(targets)[self.indices]  # Ensure numpy indexing
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        img = self.data[idx]
        candidate_labels = self.candidate_labels[idx]
        noise_target = self.targets[idx]
        original_idx = self.indices[idx]
        
        if self.transformations:
            # Ensure img is PIL Image
            if hasattr(img, 'numpy'):  # If it's a tensor
                from torchvision import transforms
                img = transforms.ToPILImage()(img)
            img = self.transformations(img)
            
        if self.return_index:
            return img, candidate_labels, noise_target, original_idx
        else:
            return img, candidate_labels, noise_target


def main(args, paths):
    # seed and device
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    init_gpuseed(args.seed, device)

    # transform
    train_transform = get_transform(dataname=args.ds, train=True)
    val_transform = get_transform(dataname=args.ds, train=False)

    # dataset
    if args.use_combined_dataset and COMBINED_DATASET_AVAILABLE:
        print("----------------Using Combined Multi-Label Dataset----------------")
        print(f"Combination ratio: {args.combination_ratio}")
        print(f"Combination method: {args.combination_method}")
        
        # Create combined dataset without transforms initially
        raw_combined_dataset, num_classes = create_combined_dataset(
            dataset_name=args.ds,
            data_root=args.data_root,
            combination_ratio=args.combination_ratio,
            combination_method=args.combination_method,
            train=True
        )
        
        # Wrap for compatibility
        ordinary_train_dataset = CombinedDatasetWrapper(raw_combined_dataset, val_transform)
        
        # Get test dataset normally (no combinations for test)
        _, test_dataset, _ = get_origin_datasets(dataname=args.ds, transform=val_transform,
                                               data_root=args.data_root)
        
        print(f"Combined dataset size: {len(ordinary_train_dataset)}")
        
    else:
        ordinary_train_dataset, test_dataset, num_classes = get_origin_datasets(dataname=args.ds, transform=val_transform,
                                                                                data_root=args.data_root)

    if args.use_combined_dataset and COMBINED_DATASET_AVAILABLE:
        # For combined dataset, generate noise labels from the combined data
        noise_labels = np.array(ordinary_train_dataset.combined_dataset.get_noise_labels(
            flip_rate=args.flip_rate, 
            seed=args.seed
        ))
    else:
        noise_labels = generate_noise_labels(train_labels=ordinary_train_dataset.targets,
                                             num_classes=num_classes,
                                             data_type=args.data_gen,
                                             flip_rate=args.flip_rate,
                                             seed=args.seed,
                                             train_data=ordinary_train_dataset.data)

    # candidate
    if args.use_combined_dataset and COMBINED_DATASET_AVAILABLE:
        print("----------------Using Combined Dataset Labels----------------")
        # Use the multi-labels from combined dataset as candidate labels
        train_candidate_labels = np.array([label for label in ordinary_train_dataset.combined_dataset.all_labels])
        print(f"Multi-labels shape: {train_candidate_labels.shape}")
        print(f"Average labels per sample: {train_candidate_labels.sum(axis=1).mean():.2f}")
    else:
        print("----------------Labeling----------------")
        prenp(args, paths, noise_labels)
        train_candidate_labels = np.load(paths['multi_labels'])
    assert (train_candidate_labels.sum(1) > 0).all()

    train_indices, val_indices = indices_split(len_dataset=len(ordinary_train_dataset),
                                               seed=args.seed,
                                               val_ratio=0.1)

    # dataloaders
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    # Handle dataset creation differently for combined vs regular datasets
    if args.use_combined_dataset and COMBINED_DATASET_AVAILABLE:
        # For combined dataset, create custom datasets directly
        val_dataset = CombinedCanTarDataset(
            combined_dataset=ordinary_train_dataset.combined_dataset,
            candidate_labels=train_candidate_labels,
            targets=noise_labels,
            indices=val_indices,
            transformations=val_transform,
            return_index=True
        )
        
        train_dataset = CombinedCanTarDataset(
            combined_dataset=ordinary_train_dataset.combined_dataset,
            candidate_labels=train_candidate_labels,
            targets=noise_labels,
            indices=train_indices,
            transformations=train_transform,
            return_index=True
        )
    else:
        # Regular dataset handling
        val_dataset = get_cantar_dataset(dataset=ordinary_train_dataset,
                                         candidate_labels=train_candidate_labels,
                                         targets=noise_labels,
                                         transformations=val_transform,
                                         indices=val_indices,
                                         return_index=True)
        train_dataset = get_cantar_dataset(dataset=ordinary_train_dataset,
                                           candidate_labels=train_candidate_labels,
                                           targets=noise_labels,
                                           transformations=train_transform,
                                           indices=train_indices,
                                           return_index=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.bs, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True,
                                             persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True,
                                               persistent_workers=True, drop_last=True)

    model = get_model(args.mo, num_classes=num_classes, fix_backbone=False)
    U_model = get_model(args.mo, num_classes=num_classes * num_classes, fix_backbone=False)

    print(paths['pre_model'])
    state = torch.load(paths['pre_model'])
    state = {k: v for k, v in state.items()}
    model.load_state_dict(state)

    model_dict = model.state_dict()
    model_dict = {k: v for k, v in model_dict.items() if ('fc' not in k)}
    U_model_dict = U_model.state_dict()
    U_model_dict.update(model_dict)
    U_model.load_state_dict(U_model_dict)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in U_model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    Ttrain_optimizer = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": U_model.parameters()}], lr=args.lr,
        weight_decay=args.wd, momentum=args.momentum,
        nesterov=args.nesterov)

    Tscheduler = get_scheduler(args.ds, Ttrain_optimizer, args.ep)

    U_model.to(device)
    model.to(device)
    model.eval()

    print("----------------Train----------------")
    if args.ds == 'clothing1m':
        b_sampler = BalancedSampler(num_classes, np.array(noise_labels[train_indices]))
    Tloss_val_best = 10000.0
    pre_acc_val = 0.0
    for epoch in range(args.pre_ep):
        if args.ds == 'clothing1m':
            sampled_indices = b_sampler.ind_unsampling()
            term_indices = np.array(train_indices)[sampled_indices].tolist()
            train_dataset = get_cantar_dataset(dataset=ordinary_train_dataset,
                                               candidate_labels=train_candidate_labels,
                                               targets=noise_labels,
                                               transformations=val_transform,
                                               indices=term_indices,
                                               return_index=True)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True,
                                                       num_workers=args.num_workers, pin_memory=True,
                                                       persistent_workers=True)
        U_model.train()
        model.train()
        for images, candidate_labels, noise_targets, indices in train_loader:
            images = images.to(device, non_blocking=True)
            noise_targets = noise_targets.to(device, non_blocking=True)
            candidate_labels = candidate_labels.to(device, non_blocking=True)
            Ttrain_optimizer.zero_grad()

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            p_tilde = probs
            l1 = F.nll_loss(torch.log(p_tilde), noise_targets)

            U = U_model(images)
            U = U.view(U.size(0), num_classes, num_classes)
            U = torch.sigmoid(U)
            U = U.float()
            p_can = torch.matmul(p_tilde.unsqueeze(1), U)
            p_can = p_can.squeeze(1)
            p_can = torch.clamp(p_can, min=0, max=1)
            l2 = F.binary_cross_entropy(p_can, candidate_labels.float())
            loss = 0.5 * (l1 + l2)

            loss.backward()
            Ttrain_optimizer.step()
        Tscheduler.step()
        U_model.eval()
        model.eval()

        with torch.no_grad():
            loss_val = 0.0
            acc_val = 0.0
            for images, candidate_labels, noise_targets, indices in val_loader:
                images = images.to(device, non_blocking=True)
                noise_targets = noise_targets.to(device, non_blocking=True)
                candidate_labels = candidate_labels.to(device, non_blocking=True)

                logits = model(images)
                probs = F.softmax(logits, dim=1)
                p_tilde = probs
                l1 = F.nll_loss(torch.log(p_tilde), noise_targets)

                _, preds = torch.max(p_tilde.data, 1)
                acc_val += (preds == noise_targets).sum().item()

                U = U_model(images)
                U = U.view(U.size(0), num_classes, num_classes)
                U = torch.sigmoid(U)
                U = U.float()
                p_can = torch.matmul(p_tilde.unsqueeze(1), U)
                p_can = p_can.squeeze(1)
                p_can = torch.clamp(p_can, min=0, max=1)
                l2 = F.binary_cross_entropy(p_can, candidate_labels.float())
                loss = 0.5 * (l1 + l2)
                loss_val += loss * len(noise_targets)

            loss_val = loss_val / len(val_dataset)
            if loss_val < Tloss_val_best:
                Tloss_val_best = loss_val
                test_accuracy_best = accuracy_check(loader=test_loader, model=model, device=device)
                best_model = deepcopy(model)
                best_U_model = deepcopy(U_model)
                print('Epoch: {}. Best_loss_val: {}.'.format(epoch, loss_val))

        if epoch == 10:
            Tloss_val_best = 10000.0
            for param in model.parameters():
                param.requires_grad = True
            for param in U_model.parameters():
                param.requires_grad = True

    model = get_model(args.mo, num_classes=num_classes, fix_backbone=False)
    model = model.to(device)
    train_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                      weight_decay=args.wd, momentum=args.momentum,
                                      nesterov=args.nesterov)
    scheduler = get_scheduler(args.ds, train_optimizer, args.ep)
    print("----------------Train_LNL----------------")
    # anchors
    probs_all = np.load(paths['pre_probs'])
    probs_all = torch.from_numpy(probs_all)
    probs_all_ = deepcopy(probs_all)
    anchors = []

    for i in range(num_classes):
        # Reference: https://github.com/xiaoboxia/T-Revision.git
        if args.filter_outlier:
            eta_thresh = np.percentile(probs_all[:, i].numpy(), 97, method='higher')
            robust_eta = probs_all[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0.0
        tk = torch.topk(probs_all[:, i], args.anchors_per_class, dim=0)
        anchors.append(tk.indices.tolist())

    # Q
    probs_all = probs_all_
    Q = torch.zeros(num_classes, num_classes)
    for i in range(num_classes):
        Q[i] = probs_all[anchors[i]].mean(0)
    Q = Q.to(device)

    # train
    val_acc_best = 0.0
    for epoch in range(args.ep):
        if args.ds == 'clothing1m':
            sampled_indices = b_sampler.ind_unsampling()
            term_indices = np.array(train_indices)[sampled_indices].tolist()
            train_dataset = get_cantar_dataset(dataset=ordinary_train_dataset,
                                               candidate_labels=train_candidate_labels,
                                               targets=noise_labels,
                                               transformations=val_transform,
                                               indices=term_indices,
                                               return_index=True)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True,
                                                       num_workers=args.num_workers, pin_memory=True,
                                                       persistent_workers=True)
        model.train()
        for images, candidate_labels, noise_targets, indices in train_loader:
            images = images.to(device, non_blocking=True)
            noise_targets = noise_targets.to(device, non_blocking=True)
            candidate_labels = candidate_labels.to(device, non_blocking=True)
            train_optimizer.zero_grad()

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            p_tilde = torch.mm(probs, Q)
            l1 = F.nll_loss(torch.log(p_tilde), noise_targets)

            U = best_U_model(images)
            U = U.to(device)
            U = U.view(U.size(0), num_classes, num_classes)
            U = torch.sigmoid(U)
            U = U.float()
            p_can = torch.matmul(p_tilde.unsqueeze(1), U)
            p_can = p_can.squeeze(1)
            p_can = torch.clamp(p_can, min=0, max=1)

            l2 = F.binary_cross_entropy(p_can, candidate_labels.float())
            loss = 0.5 * (l1 + l2)

            loss.backward()
            train_optimizer.step()
        scheduler.step()
        model.eval()

        with torch.no_grad():
            acc_val = 0.0
            for images, candidate_labels, noise_targets, indices in val_loader:
                images = images.to(device, non_blocking=True)
                noise_targets = noise_targets.to(device, non_blocking=True)

                logits = model(images)
                probs = F.softmax(logits, dim=1)
                p_tilde = torch.mm(probs, Q)
                _, preds = torch.max(p_tilde.data, 1)
                acc_val += (preds == noise_targets).sum().item()

            acc_val = acc_val / len(val_dataset)
            if acc_val >= val_acc_best:
                val_acc_best = acc_val
                best_model = deepcopy(model)
                test_accuracy_best = accuracy_check(loader=test_loader, model=model, device=device)
                print('Epoch: {}. Best_ACC: {}.'.format(epoch, test_accuracy_best))
    torch.save(best_model.state_dict(), paths['best_model'])
    torch.save(best_U_model.state_dict(), paths['U_model'])
    print("Test ACC:", test_accuracy_best)
    print("Test Final:", accuracy_check(loader=test_loader, model=model, device=device))


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='optimizer\'s learning rate', default=0.01, type=float,
                        required=False)
    parser.add_argument('--wd', help='weight decay', default=1e-2, type=float,
                        required=False)
    parser.add_argument('--momentum', help='momentum of opt', default=0.9, type=float, required=False)
    parser.add_argument('--nesterov', help='nesterov or not', action='store_true')
    parser.add_argument('--bs', help='batch_size of ordinary labels.', default=128, type=int,
                        required=False)
    parser.add_argument('--ep', help='number of epochs', type=int, default=50,
                        required=False)
    parser.add_argument('--ds', help='specify a dataset', default='cifar-10', type=str,
                        choices=['mnist', 'cifar-10', 'cifar-100', 'clothing1m'],
                        required=False)
    parser.add_argument('--mo', help='models name', default='resnet34',
                        choices=['resnet', 'resnet34', 'resnet50', 'lenet'],
                        type=str, required=False)
    parser.add_argument('--data_gen', help='data generate strategy', default='pair',
                        choices=['symmetry', 'pair', 'idn'],
                        type=str, required=False)
    parser.add_argument('--gpu', help='used gpu id', default='0', type=str, required=False)
    parser.add_argument('--flip_rate', help='noise flip rate', type=float, default=0.4, required=False)
    parser.add_argument('--seed', help='Random seed', default=40, type=int, required=False)
    parser.add_argument('--crop_ratio', help='crop ratio', type=float, default=0.8, required=False)
    parser.add_argument('--save_dir', help='results dir', default='./res', type=str, required=False)
    parser.add_argument('--data_root', help='data dir', default='~/data', type=str, required=False)
    parser.add_argument('--num_workers', help='num worker', default=4, type=int, required=False)
    parser.add_argument('--use_attention_crop', help='use attention-based cropping instead of random FiveCrop', 
                        action='store_true')
    
    # Combined dataset arguments
    parser.add_argument('--use_combined_dataset', help='use combined multi-label dataset instead of PLM labeling',
                        action='store_true')
    parser.add_argument('--combination_ratio', help='ratio of combined samples vs original samples',
                        type=float, default=0.3)
    parser.add_argument('--combination_method', help='method for combining images',
                        choices=['horizontal', 'vertical', 'blend', 'grid'], default='horizontal')
    args = parser.parse_args()

    if args.ds == 'cifar-10':
        args.lr = 0.01
        args.wd = 1e-2
        args.ep = 20
        args.pre_ep = 20
        args.crop_ratio = 0.8
        if args.data_gen == 'idn':
            args.mo = 'resnet34'
        else:
            args.mo = 'resnet'
        args.filter_outlier = True
        args.anchors_per_class = 10

    if args.ds == 'cifar-100':
        args.lr = 0.05
        args.wd = 1e-3
        args.ep = 80
        args.pre_ep = 80
        args.crop_ratio = 0.8
        args.mo = 'resnet34'
        args.filter_outlier = False
        args.anchors_per_class = 10

    if args.ds == 'mnist':
        args.lr = 0.05
        args.wd = 1e-4
        args.ep = 50
        args.pre_ep = 50
        args.crop_ratio = 0.8
        args.mo = 'lenet'
        args.filter_outlier = True
        args.anchors_per_class = 10

    if args.ds == 'clothing1m':
        args.lr = 0.01
        args.wd = 1e-3
        args.bs = 32
        args.ep = 15
        args.pre_ep = 15
        args.crop_ratio = 0.7
        args.mo = 'resnet50'
        args.filter_outlier = True
        args.anchors_per_class = 10

    paths = get_paths(args)
    start_time = time.time()
    main(args, paths)
    end_time = time.time()
    print("time: ", end_time - start_time)
