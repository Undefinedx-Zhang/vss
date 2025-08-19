"""
FGP Core Implementation
Adapted from: https://github.com/FGP-code/FGP

Key concepts:
- Feature-Gradient correlation for channel importance
- Multi-class gradient analysis
- Channel support value computation across different inputs
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class FeatureGradientAnalyzer:
    """Analyzes feature maps and gradients to determine channel importance"""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.hooks = []
    
    def register_hooks(self, model: nn.Module, target_layers: List[nn.Module]):
        """Register forward and backward hooks to capture features and gradients"""
        
        def forward_hook(name):
            def hook(module, input, output):
                # Store feature maps
                self.feature_maps[name] = output.detach().clone()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                # Store gradients
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach().clone()
            return hook
        
        # Register hooks for target layers
        for i, layer in enumerate(target_layers):
            layer_name = f"layer_{i}"
            fhook = layer.register_forward_hook(forward_hook(layer_name))
            bhook = layer.register_backward_hook(backward_hook(layer_name))
            self.hooks.extend([fhook, bhook])
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def compute_channel_support(self, model: nn.Module, inputs: torch.Tensor, 
                               targets: torch.Tensor, target_layers: List[nn.Module]) -> Dict[str, torch.Tensor]:
        """
        Compute channel support values based on feature-gradient correlation
        
        Args:
            model: The neural network model
            inputs: Input batch [N, C, H, W]
            targets: Target labels [N]
            target_layers: List of Conv2d layers to analyze
        
        Returns:
            Dictionary mapping layer names to channel support scores
        """
        model.train()
        self.register_hooks(model, target_layers)
        
        channel_supports = {}
        
        try:
            # Forward pass
            outputs = model(inputs)
            # Handle segmentation outputs: dict or 4D logits
            if isinstance(outputs, dict):
                # Prefer key 'out' if available
                outputs = outputs.get('out', next(iter(outputs.values())))
            if outputs.dim() == 4:
                # [N, C, H, W] -> [N, C] for CE
                outputs_for_ce = outputs.mean(dim=(2, 3))
            else:
                outputs_for_ce = outputs

            # Compute loss for each class to get diverse gradients
            support_scores = {}
            
            batch_size = inputs.size(0)
            num_classes = self.num_classes
            for class_id in range(num_classes):
                # Create per-sample class targets [N]
                class_targets = torch.full((batch_size,), class_id, device=outputs_for_ce.device, dtype=torch.long)
                loss = F.cross_entropy(outputs_for_ce, class_targets)
                
                # Backward pass
                model.zero_grad()
                loss.backward(retain_graph=True)
                
                # Compute feature-gradient correlation for each layer
                for layer_idx, layer_name in enumerate([f"layer_{i}" for i in range(len(target_layers))]):
                    if layer_name in self.feature_maps and layer_name in self.gradients:
                        features = self.feature_maps[layer_name]  # [N, C, H, W]
                        gradients = self.gradients[layer_name]    # [N, C, H, W]
                        
                        # Compute channel-wise feature-gradient correlation
                        # Average over spatial dimensions and batch
                        feat_avg = features.mean(dim=(0, 2, 3))  # [C]
                        grad_avg = gradients.mean(dim=(0, 2, 3))  # [C]
                        
                        # Correlation-based importance (element-wise product)
                        channel_importance = torch.abs(feat_avg * grad_avg)
                        
                        if layer_name not in support_scores:
                            support_scores[layer_name] = []
                        support_scores[layer_name].append(channel_importance)
            
            # Aggregate support scores across classes
            for layer_name, class_scores in support_scores.items():
                # Stack scores from all classes and compute statistics
                stacked_scores = torch.stack(class_scores, dim=0)  # [num_classes, C]
                
                # Support value: channels that are consistently important across classes
                # Use minimum to find channels that are important for ALL classes
                min_scores = torch.min(stacked_scores, dim=0)[0]
                # Use mean to balance between consistency and magnitude
                mean_scores = torch.mean(stacked_scores, dim=0)
                
                # Combine min and mean for robust support measure
                channel_supports[layer_name] = 0.7 * mean_scores + 0.3 * min_scores
        
        finally:
            self.remove_hooks()
            self.feature_maps.clear()
            self.gradients.clear()
        
        return channel_supports


def compute_fgp_importance(model: nn.Module, data_loader, device: str, num_samples: int = 64) -> Dict[nn.Module, torch.Tensor]:
    """
    Compute FGP-based channel importance scores
    
    Args:
        model: Neural network model
        data_loader: DataLoader with training data
        device: Device to run computation on
        num_samples: Number of samples to use for analysis
    
    Returns:
        Dictionary mapping Conv2d modules to importance scores
    """
    model.to(device)
    analyzer = FeatureGradientAnalyzer()
    
    # Collect target Conv2d layers
    target_layers = []
    layer_to_module = {}
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            layer_idx = len(target_layers)
            target_layers.append(module)
            layer_to_module[f"layer_{layer_idx}"] = module
    
    if not target_layers:
        return {}
    
    # Collect samples for analysis
    all_inputs = []
    all_targets = []
    sample_count = 0
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        all_inputs.append(inputs)
        all_targets.append(targets)
        sample_count += inputs.size(0)
        
        if sample_count >= num_samples:
            break
    
    if not all_inputs:
        return {module: torch.zeros(module.out_channels, device=device) for module in target_layers}
    
    # Concatenate all samples
    inputs_batch = torch.cat(all_inputs, dim=0)[:num_samples]
    targets_batch = torch.cat(all_targets, dim=0)[:num_samples]
    
    # Compute channel support values
    channel_supports = analyzer.compute_channel_support(
        model, inputs_batch, targets_batch, target_layers
    )
    
    # Map back to module dictionary
    importance_scores = {}
    for layer_name, support_scores in channel_supports.items():
        if layer_name in layer_to_module:
            module = layer_to_module[layer_name]
            importance_scores[module] = support_scores
    
    return importance_scores


def visualize_channel_importance(feature_maps: torch.Tensor, importance_scores: torch.Tensor, 
                                save_path: str = None) -> np.ndarray:
    """
    Visualize channel importance using heatmaps
    
    Args:
        feature_maps: Feature maps [C, H, W] 
        importance_scores: Channel importance scores [C]
        save_path: Optional path to save visualization
    
    Returns:
        Visualization as numpy array
    """
    import matplotlib.pyplot as plt
    
    num_channels = feature_maps.size(0)
    top_k = min(8, num_channels)  # Show top 8 channels
    
    # Get top-k most important channels
    _, top_indices = torch.topk(importance_scores, top_k)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(top_indices):
        if i >= 8:
            break
        
        channel_map = feature_maps[channel_idx].cpu().numpy()
        im = axes[i].imshow(channel_map, cmap='viridis')
        axes[i].set_title(f'Ch {channel_idx.item()}: {importance_scores[channel_idx]:.3f}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img
