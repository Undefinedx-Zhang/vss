"""
Training and Evaluation utilities for Semantic Segmentation
"""
from __future__ import annotations
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SegmentationMetrics


class SegmentationLoss(nn.Module):
    """Combined loss for segmentation"""
    
    def __init__(self, ignore_index: int = 255, weight: Optional[torch.Tensor] = None, 
                 aux_weight: float = 0.4):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weight)
        self.aux_weight = aux_weight
    
    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            # Handle auxiliary outputs (e.g., from DeepLabV3)
            main_loss = self.ce_loss(outputs['out'], targets)
            if 'aux' in outputs:
                aux_loss = self.ce_loss(outputs['aux'], targets)
                return main_loss + self.aux_weight * aux_loss
            return main_loss
        else:
            return self.ce_loss(outputs, targets)


def train_one_epoch_seg(model: nn.Module, loader: DataLoader, criterion, optimizer, 
                       device: str, num_classes: int, ignore_index: int = 255):
    """Train one epoch for segmentation"""
    model.train()
    running_loss = 0.0
    metrics = SegmentationMetrics(num_classes, ignore_index)
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update metrics
        if isinstance(outputs, dict):
            pred = outputs['out']
        else:
            pred = outputs
        metrics.update(pred.detach(), targets)
        
        # Update progress bar
        current_results = metrics.get_results()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mIoU': f"{current_results['mIoU']:.3f}",
            'PixAcc': f"{current_results['Pixel_Accuracy']:.3f}"
        })
    
    results = metrics.get_results()
    results['loss'] = running_loss / len(loader)
    return results


@torch.no_grad()
def evaluate_seg(model: nn.Module, loader: DataLoader, criterion, device: str,
                num_classes: int, ignore_index: int = 255):
    """Evaluate segmentation model"""
    model.eval()
    running_loss = 0.0
    metrics = SegmentationMetrics(num_classes, ignore_index)
    
    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        
        # Update metrics
        if isinstance(outputs, dict):
            pred = outputs['out']
        else:
            pred = outputs
        metrics.update(pred, targets)
        
        # Update progress bar
        current_results = metrics.get_results()
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mIoU': f"{current_results['mIoU']:.3f}",
            'PixAcc': f"{current_results['Pixel_Accuracy']:.3f}"
        })
    
    results = metrics.get_results()
    results['loss'] = running_loss / len(loader)
    return results


def fit_seg(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
           epochs: int, lr: float, weight_decay: float, device: str, save_dir: str,
           num_classes: int, ignore_index: int = 255, scheduler_type: str = 'poly'):
    """Fit segmentation model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss function
    criterion = SegmentationLoss(ignore_index=ignore_index)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    
    # Scheduler
    if scheduler_type == 'poly':
        scheduler = PolynomialLR(optimizer, total_iters=epochs, power=0.9)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[epochs//2, int(epochs*0.75)], gamma=0.1)
    
    best_miou = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        train_results = train_one_epoch_seg(
            model, train_loader, criterion, optimizer, device, num_classes, ignore_index
        )
        
        # Validation
        val_results = evaluate_seg(
            model, val_loader, criterion, device, num_classes, ignore_index
        )
        
        scheduler.step()
        
        # Save results
        train_history.append(train_results)
        val_history.append(val_results)
        
        print(f"Train - Loss: {train_results['loss']:.4f}, mIoU: {train_results['mIoU']:.3f}, "
              f"PixAcc: {train_results['Pixel_Accuracy']:.3f}")
        print(f"Val   - Loss: {val_results['loss']:.4f}, mIoU: {val_results['mIoU']:.3f}, "
              f"PixAcc: {val_results['Pixel_Accuracy']:.3f}")
        
        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_miou": val_results['mIoU'],
            "train_history": train_history,
            "val_history": val_history
        }
        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        
        # Save best model
        if val_results['mIoU'] > best_miou:
            best_miou = val_results['mIoU']
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            print(f"New best mIoU: {best_miou:.3f}")
    
    return best_miou, train_history, val_history


def load_seg_checkpoint(model: nn.Module, ckpt_path: str, strict: bool = True):
    """Load segmentation model checkpoint"""
    state = torch.load(ckpt_path, map_location="cpu")
    if "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    
    model.load_state_dict(state_dict, strict=strict)
    return model


class EarlyStopping:
    """Early stopping utility for segmentation training"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()


def compute_class_weights(loader: DataLoader, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """Compute class weights for balanced training"""
    print("Computing class weights...")
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    for _, targets in tqdm(loader, desc="Computing weights"):
        mask = (targets != ignore_index)
        valid_targets = targets[mask]
        total_pixels += valid_targets.numel()
        
        for c in range(num_classes):
            class_counts[c] += (valid_targets == c).sum().item()
    
    # Compute inverse frequency weights
    class_frequencies = class_counts / total_pixels
    class_weights = 1.0 / (class_frequencies + 1e-8)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print(f"Class frequencies: {class_frequencies}")
    print(f"Class weights: {class_weights}")
    
    return class_weights
