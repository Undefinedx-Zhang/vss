"""
Semantic Segmentation Metrics
Includes mIoU, pixel accuracy, and other segmentation-specific metrics
"""
from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from thop import profile, clever_format


class SegmentationMetrics:
    """Comprehensive segmentation metrics calculator"""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Update metrics with new predictions and targets
        
        Args:
            pred: Predictions [N, H, W] or [N, C, H, W]
            target: Ground truth [N, H, W]
        """
        if pred.dim() == 4:  # [N, C, H, W]
            pred = pred.argmax(dim=1)  # [N, H, W]
        
        pred = pred.flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()
        
        # Remove ignored pixels
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        # Update confusion matrix
        for t, p in zip(target, pred):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1
        
        self.total_samples += len(target)
    
    def compute_iou(self) -> np.ndarray:
        """Compute IoU for each class"""
        iou = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + 
            np.sum(self.confusion_matrix, axis=0) - 
            np.diag(self.confusion_matrix) + 1e-8
        )
        return iou
    
    def compute_miou(self) -> float:
        """Compute mean IoU"""
        iou = self.compute_iou()
        return np.nanmean(iou)
    
    def compute_pixel_accuracy(self) -> float:
        """Compute pixel accuracy"""
        if self.total_samples == 0:
            return 0.0
        acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-8)
        return acc
    
    def compute_mean_accuracy(self) -> float:
        """Compute mean class accuracy"""
        acc_per_class = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-8)
        return np.nanmean(acc_per_class)
    
    def compute_frequency_weighted_iou(self) -> float:
        """Compute frequency weighted IoU"""
        iou = self.compute_iou()
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix) + 1e-8)
        return np.sum(freq * iou)
    
    def get_results(self) -> Dict[str, float]:
        """Get all computed metrics"""
        return {
            'mIoU': self.compute_miou(),
            'Pixel_Accuracy': self.compute_pixel_accuracy(),
            'Mean_Accuracy': self.compute_mean_accuracy(),
            'Frequency_Weighted_IoU': self.compute_frequency_weighted_iou()
        }
    
    def get_class_results(self, class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get per-class results"""
        iou = self.compute_iou()
        acc_per_class = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-8)
        
        results = {}
        for i in range(self.num_classes):
            class_name = class_names[i] if class_names else f"Class_{i}"
            results[f"{class_name}_IoU"] = iou[i]
            results[f"{class_name}_Acc"] = acc_per_class[i]
        
        return results


@torch.no_grad()
def evaluate_segmentation(model: nn.Module, loader: DataLoader, device: str, 
                         num_classes: int, ignore_index: int = 255) -> Dict[str, float]:
    """Evaluate segmentation model"""
    model.eval()
    metrics = SegmentationMetrics(num_classes, ignore_index)
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(images)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            outputs = outputs['out']
        
        metrics.update(outputs, targets)
    
    return metrics.get_results()


def speed_benchmark_seg(model: nn.Module, input_size: Tuple[int, int, int, int], 
                       device: str) -> Dict[str, float]:
    """Benchmark segmentation model speed"""
    model.eval()
    x = torch.randn(*input_size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    iters = 50
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t1 = time.time()
    elapsed = (t1 - t0) / iters
    
    return {
        "avg_ms_per_iter": elapsed * 1000.0,
        "fps": input_size[0] / elapsed,
        "pixels_per_second": (input_size[0] * input_size[2] * input_size[3]) / elapsed
    }


def flops_params_seg(model: nn.Module, input_size: Tuple[int, int, int, int]) -> Dict[str, str]:
    """Compute FLOPs and parameters for segmentation model"""
    # Ensure the dummy input is created on the same device as the model
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    x = torch.randn(*input_size, device=model_device)

    try:
        flops, params = profile(model, inputs=(x,), verbose=False)
        flops_c, params_c = clever_format([flops, params], "%.3f")
        return {
            "FLOPs": flops_c,
            "Params": params_c,
            "FLOPs_raw": flops,
            "Params_raw": params,
        }
    except Exception as e:
        print(f"Failed to compute FLOPs/Params: {e}")
        total_params = sum(p.numel() for p in model.parameters())
        return {
            "FLOPs": "N/A",
            "Params": f"{total_params/1e6:.2f}M",
            "FLOPs_raw": 0,
            "Params_raw": total_params,
        }


def oracle_channel_importance_seg(model: nn.Module, layer: nn.Conv2d, loader: DataLoader, 
                                 device: str, num_classes: int, ignore_index: int = 255,
                                 max_samples: int = 32) -> torch.Tensor:
    """
    Compute oracle channel importance for segmentation based on mIoU drop
    
    Args:
        model: Segmentation model
        layer: Target convolutional layer
        loader: Data loader
        device: Device
        num_classes: Number of segmentation classes
        ignore_index: Index to ignore in evaluation
        max_samples: Maximum samples to use
    
    Returns:
        Channel importance scores (higher = more important)
    """
    model.eval()
    
    # Collect a small batch for analysis
    images_list = []
    targets_list = []
    sample_count = 0
    
    for images, targets in loader:
        images_list.append(images.to(device))
        targets_list.append(targets.to(device))
        sample_count += images.size(0)
        if sample_count >= max_samples:
            break
    
    if not images_list:
        return torch.zeros(layer.out_channels, device=device)
    
    images_batch = torch.cat(images_list, dim=0)[:max_samples]
    targets_batch = torch.cat(targets_list, dim=0)[:max_samples]
    
    # Compute baseline mIoU
    with torch.no_grad():
        base_outputs = model(images_batch)
        if isinstance(base_outputs, dict):
            base_outputs = base_outputs['out']
        
        base_metrics = SegmentationMetrics(num_classes, ignore_index)
        base_metrics.update(base_outputs, targets_batch)
        base_miou = base_metrics.compute_miou()
    
    out_ch = layer.out_channels
    importance_scores = torch.zeros(out_ch, device=device)
    
    # Backup original weights
    weight_backup = layer.weight.data.clone()
    bias_backup = layer.bias.data.clone() if layer.bias is not None else None
    
    for c in range(out_ch):
        # Zero out channel c
        layer.weight.data[c].zero_()
        if layer.bias is not None:
            layer.bias.data[c] = 0.0
        
        with torch.no_grad():
            outputs = model(images_batch)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            
            metrics = SegmentationMetrics(num_classes, ignore_index)
            metrics.update(outputs, targets_batch)
            current_miou = metrics.compute_miou()
        
        # Importance = drop in mIoU when channel is removed
        importance_scores[c] = max(0, base_miou - current_miou)
        
        # Restore weights
        layer.weight.data.copy_(weight_backup)
        if layer.bias is not None:
            layer.bias.data.copy_(bias_backup)
    
    return importance_scores


def reconstruction_error_seg(model_a: nn.Module, model_b: nn.Module, loader: DataLoader, 
                            device: str, max_batches: int = 3) -> float:
    """Compute reconstruction error between two segmentation models"""
    model_a.eval()
    model_b.eval()
    
    total_mse = 0.0
    count = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= max_batches:
                break
                
            images = images.to(device)
            
            out_a = model_a(images)
            out_b = model_b(images)
            
            # Handle dict outputs
            if isinstance(out_a, dict):
                out_a = out_a['out']
            if isinstance(out_b, dict):
                out_b = out_b['out']
            
            # Compute MSE on logits
            mse = F.mse_loss(out_a, out_b).item()
            total_mse += mse
            count += 1
    
    return total_mse / max(1, count)


def compute_seg_prune_ratio(model_before: nn.Module, model_after: nn.Module) -> Dict[str, float]:
    """Compute pruning ratios for segmentation model"""
    def count_conv_channels(model):
        total_channels = 0
        conv_layers = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                total_channels += m.out_channels
                conv_layers += 1
        return total_channels, conv_layers
    
    before_channels, before_layers = count_conv_channels(model_before)
    after_channels, after_layers = count_conv_channels(model_after)
    
    channel_ratio = (before_channels - after_channels) / max(1, before_channels)
    layer_ratio = (before_layers - after_layers) / max(1, before_layers)
    
    return {
        "channel_prune_ratio": channel_ratio,
        "layer_prune_ratio": layer_ratio,
        "channels_before": before_channels,
        "channels_after": after_channels,
        "layers_before": before_layers,
        "layers_after": after_layers
    }


def visualize_segmentation_results(images: torch.Tensor, targets: torch.Tensor, 
                                  predictions: torch.Tensor, class_colors: List[List[int]],
                                  save_path: Optional[str] = None) -> np.ndarray:
    """Visualize segmentation results"""
    import matplotlib.pyplot as plt
    
    batch_size = min(4, images.size(0))
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, batch_size * 4))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize images (assuming ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    for i in range(batch_size):
        # Original image
        img = images_denorm[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        target_colored = colorize_mask(targets[i].cpu().numpy(), class_colors)
        axes[i, 1].imshow(target_colored)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_colored = colorize_mask(predictions[i].cpu().numpy(), class_colors)
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img_array


def colorize_mask(mask: np.ndarray, colors: List[List[int]]) -> np.ndarray:
    """Convert segmentation mask to colored image"""
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in enumerate(colors):
        colored_mask[mask == class_id] = color
    
    return colored_mask


def save_seg_report(path: str, data: Dict):
    """Save segmentation evaluation report"""
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
