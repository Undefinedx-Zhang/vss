"""
SIRFP Core Implementation
Adapted from: https://github.com/dywu98/SIRFP.git

Key concepts:
- Spatial-aware Information Redundancy analysis
- Channel importance based on spatial information content
- Multi-scale spatial feature analysis
- Redundancy detection through spatial correlation
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


class SpatialInformationAnalyzer:
    """Analyzes spatial information content in feature maps"""
    
    def __init__(self):
        self.feature_maps: Dict[str, torch.Tensor] = {}
        self.hooks = []
    
    def register_hooks(self, model: nn.Module, target_layers: List[nn.Module]):
        """Register forward hooks to capture feature maps"""
        
        def forward_hook(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach().clone()
            return hook
        
        for i, layer in enumerate(target_layers):
            layer_name = f"layer_{i}"
            hook = layer.register_forward_hook(forward_hook(layer_name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def compute_spatial_information_entropy(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial information entropy for each channel
        
        Args:
            feature_map: Feature map tensor [N, C, H, W]
        
        Returns:
            Entropy values for each channel [C]
        """
        N, C, H, W = feature_map.shape
        entropies = torch.zeros(C, device=feature_map.device)
        
        for c in range(C):
            channel_map = feature_map[:, c, :, :].flatten(1)  # [N, H*W]
            
            # Normalize to probability distribution
            channel_map = torch.abs(channel_map)
            channel_sum = channel_map.sum(dim=1, keepdim=True)
            channel_sum = torch.clamp(channel_sum, min=1e-8)
            prob_map = channel_map / channel_sum  # [N, H*W]
            
            # Compute entropy
            log_prob = torch.log(prob_map + 1e-8)
            entropy = -(prob_map * log_prob).sum(dim=1).mean()  # Average over batch
            entropies[c] = entropy
        
        return entropies
    
    def compute_spatial_correlation(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial correlation between channels
        
        Args:
            feature_map: Feature map tensor [N, C, H, W]
        
        Returns:
            Correlation matrix [C, C]
        """
        N, C, H, W = feature_map.shape

        # Downsample spatial resolution to cap compute cost
        max_hw = 32
        if H > max_hw or W > max_hw:
            new_size = (min(H, max_hw), min(W, max_hw))
            feature_map = F.interpolate(feature_map, size=new_size, mode='bilinear', align_corners=False)
            N, C, H, W = feature_map.shape

        # Vectorized correlation across channels
        # Shape to [C, N*H*W]
        X = feature_map.view(N, C, -1).permute(1, 0, 2).contiguous().view(C, -1)
        X = X - X.mean(dim=1, keepdim=True)
        std = X.std(dim=1, keepdim=True) + 1e-8
        X_norm = X / std
        # Correlation matrix approximation via normalized inner product
        correlations = (X_norm @ X_norm.t()) / X_norm.size(1)
        correlations = torch.clamp(correlations, -1.0, 1.0).abs()
        return correlations
    
    def compute_multiscale_spatial_features(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale spatial features for redundancy analysis
        
        Args:
            feature_map: Feature map tensor [N, C, H, W]
        
        Returns:
            Dictionary of spatial features at different scales
        """
        features = {}
        
        # Original scale
        features['entropy'] = self.compute_spatial_information_entropy(feature_map)
        features['correlation'] = self.compute_spatial_correlation(feature_map)
        
        # Downsampled scales for multi-scale analysis
        for scale in [0.5, 0.25]:
            if feature_map.size(2) * scale >= 2 and feature_map.size(3) * scale >= 2:
                new_size = (int(feature_map.size(2) * scale), int(feature_map.size(3) * scale))
                downsampled = F.interpolate(feature_map, size=new_size, mode='bilinear', align_corners=False)
                
                scale_key = f'entropy_scale_{scale}'
                features[scale_key] = self.compute_spatial_information_entropy(downsampled)
        
        return features


def compute_sirfp_importance(model: nn.Module, data_loader, device: str, num_samples: int = 32) -> Dict[nn.Module, torch.Tensor]:
    """
    Compute SIRFP-based channel importance scores
    
    Args:
        model: Neural network model
        data_loader: DataLoader with data samples
        device: Device to run computation on
        num_samples: Number of samples to use for analysis
    
    Returns:
        Dictionary mapping Conv2d modules to importance scores
    """
    model.to(device)
    model.eval()
    
    analyzer = SpatialInformationAnalyzer()
    
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
    
    analyzer.register_hooks(model, target_layers)
    
    try:
        # Collect feature maps from multiple samples
        all_features = {f"layer_{i}": [] for i in range(len(target_layers))}
        
        sample_count = 0
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                
                # Forward pass to collect features
                _ = model(inputs)
                
                # Store feature maps
                for layer_name in all_features.keys():
                    if layer_name in analyzer.feature_maps:
                        all_features[layer_name].append(analyzer.feature_maps[layer_name])
                
                sample_count += inputs.size(0)
                if sample_count >= num_samples:
                    break
        
        # Compute importance scores for each layer
        importance_scores = {}
        
        for layer_name, feature_list in all_features.items():
            if not feature_list:
                continue
            
            # Concatenate features from all samples
            combined_features = torch.cat(feature_list, dim=0)
            
            # Compute spatial information features
            spatial_features = analyzer.compute_multiscale_spatial_features(combined_features)
            
            # Compute channel importance based on spatial information
            entropy_scores = spatial_features['entropy']
            correlation_matrix = spatial_features['correlation']
            
            # Importance = high entropy (information content) + low correlation (uniqueness)
            # Average correlation with other channels (exclude diagonal)
            C = correlation_matrix.size(0)
            row_sum = correlation_matrix.sum(dim=1) - torch.diag(correlation_matrix)
            denom = max(C - 1, 1)
            avg_correlation = row_sum / denom
            
            # Combine entropy and uniqueness (inverse correlation)
            importance = entropy_scores * (1.0 - torch.clamp(avg_correlation, 0, 1))
            
            # Add multi-scale information if available
            for scale in [0.5, 0.25]:
                scale_key = f'entropy_scale_{scale}'
                if scale_key in spatial_features:
                    importance += 0.3 * spatial_features[scale_key]  # Weight multi-scale features
            
            # Map back to module
            if layer_name in layer_to_module:
                module = layer_to_module[layer_name]
                importance_scores[module] = importance
    
    finally:
        analyzer.remove_hooks()
    
    return importance_scores


def compute_spatial_redundancy_matrix(feature_maps: torch.Tensor, threshold: float = 0.8) -> torch.Tensor:
    """
    Compute spatial redundancy matrix between channels
    
    Args:
        feature_maps: Feature maps [N, C, H, W]
        threshold: Correlation threshold for redundancy
    
    Returns:
        Binary redundancy matrix [C, C]
    """
    analyzer = SpatialInformationAnalyzer()
    correlation_matrix = analyzer.compute_spatial_correlation(feature_maps)
    
    # Create redundancy matrix (1 if channels are redundant, 0 otherwise)
    redundancy_matrix = (correlation_matrix > threshold).float()
    
    # Remove self-correlation
    redundancy_matrix.fill_diagonal_(0)
    
    return redundancy_matrix


def visualize_spatial_information(feature_maps: torch.Tensor, importance_scores: torch.Tensor, 
                                 save_path: str = None) -> np.ndarray:
    """
    Visualize spatial information patterns in feature maps
    
    Args:
        feature_maps: Feature maps [C, H, W]
        importance_scores: Channel importance scores [C]
        save_path: Optional path to save visualization
    
    Returns:
        Visualization as numpy array
    """
    import matplotlib.pyplot as plt
    
    num_channels = feature_maps.size(0)
    top_k = min(6, num_channels)
    
    # Get top-k most important channels
    _, top_indices = torch.topk(importance_scores, top_k)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(top_indices):
        if i >= 6:
            break
        
        channel_map = feature_maps[channel_idx].cpu().numpy()
        
        # Compute spatial information visualization
        im = axes[i].imshow(channel_map, cmap='plasma', interpolation='bilinear')
        axes[i].set_title(f'Ch {channel_idx.item()}: Spatial Info {importance_scores[channel_idx]:.3f}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle('SIRFP: Spatial Information Analysis')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return img
