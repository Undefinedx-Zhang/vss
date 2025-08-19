"""
SIRFP Pruner Implementation
Integrates SIRFP with our unified pruning interface
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import Pruner, PruningConfig
from .sirfp_core import compute_sirfp_importance


class SIRFPPruner(Pruner):
    """
    SIRFP (Structural Pruning via Spatial-aware Information Redundancy) implementation
    
    Identifies channel redundancy based on spatial information patterns
    and multi-scale spatial feature analysis
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        super().__init__(model, config)
        self.num_samples = getattr(config, 'sirfp_samples', 32)
        self.correlation_threshold = getattr(config, 'sirfp_corr_threshold', 0.8)
    
    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        """
        Compute SIRFP-based importance scores using spatial information analysis
        
        Args:
            example_inputs: Example input tensor for the model
        
        Returns:
            Dictionary mapping Conv2d modules to their channel importance scores
        """
        device = example_inputs.device
        
        # Create a simple data loader from example inputs
        batch_size = example_inputs.size(0)

        class SimpleDataset:
            def __init__(self, inputs):
                self.inputs = inputs
            
            def __len__(self):
                return self.inputs.size(0)
            
            def __getitem__(self, idx):
                return self.inputs[idx], torch.tensor(0)  # Dummy target per-sample

        dataset = SimpleDataset(example_inputs)
        data_loader = DataLoader(dataset, batch_size=min(batch_size, self.num_samples), shuffle=False)
        
        # Compute SIRFP importance scores
        try:
            importance_scores = compute_sirfp_importance(
                self.model, 
                data_loader, 
                device, 
                num_samples=min(self.num_samples, batch_size)
            )
        except Exception as e:
            print(f"SIRFP importance computation failed: {e}")
            # Fallback: do not prune if importance cannot be computed
            importance_scores = {}
        
        return importance_scores
    
    def importance_scores_with_data(self, data_loader: DataLoader) -> Dict[nn.Module, torch.Tensor]:
        """
        Compute SIRFP importance scores using actual training data
        
        Args:
            data_loader: Training data loader
        
        Returns:
            Dictionary mapping Conv2d modules to their channel importance scores
        """
        device = next(self.model.parameters()).device
        
        try:
            importance_scores = compute_sirfp_importance(
                self.model, 
                data_loader, 
                device, 
                num_samples=self.num_samples
            )
        except Exception as e:
            print(f"SIRFP importance computation failed: {e}")
            # Fallback to uniform scores
            importance_scores = {}
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                    importance_scores[module] = torch.ones(module.out_channels, device=device)
        
        return importance_scores
    
    def analyze_spatial_redundancy(self, data_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Analyze spatial redundancy patterns in the model
        
        Args:
            data_loader: Data loader for analysis
        
        Returns:
            Dictionary containing redundancy analysis results
        """
        from .sirfp_core import compute_spatial_redundancy_matrix, SpatialInformationAnalyzer
        
        device = next(self.model.parameters()).device
        self.model.eval()
        
        analyzer = SpatialInformationAnalyzer()
        target_layers = [m for m in self.model.modules() if isinstance(m, nn.Conv2d) and m.out_channels > 1]
        
        if not target_layers:
            return {}
        
        analyzer.register_hooks(self.model, target_layers)
        
        redundancy_analysis = {}
        
        try:
            with torch.no_grad():
                for inputs, _ in data_loader:
                    inputs = inputs.to(device)
                    _ = self.model(inputs)
                    
                    # Analyze redundancy for each layer
                    for i, layer_name in enumerate([f"layer_{j}" for j in range(len(target_layers))]):
                        if layer_name in analyzer.feature_maps:
                            feature_maps = analyzer.feature_maps[layer_name]
                            redundancy_matrix = compute_spatial_redundancy_matrix(
                                feature_maps, self.correlation_threshold
                            )
                            redundancy_analysis[f"layer_{i}_redundancy"] = redundancy_matrix
                    
                    break  # Analyze only first batch
        
        finally:
            analyzer.remove_hooks()
        
        return redundancy_analysis
