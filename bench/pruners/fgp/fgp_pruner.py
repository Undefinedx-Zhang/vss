"""
FGP Pruner Implementation
Integrates FGP with our unified pruning interface
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import Pruner, PruningConfig
from .fgp_core import compute_fgp_importance


class FGPPruner(Pruner):
    """
    FGP (Feature-Gradient-Prune) implementation
    
    Uses feature maps and gradients to identify important channels
    based on their correlation and support across different classes
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        super().__init__(model, config)
        self.num_samples = getattr(config, 'fgp_samples', 64)
        self.num_classes = 10  # Default for CIFAR-10
    
    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        """
        Compute FGP-based importance scores using feature-gradient correlation
        
        Args:
            example_inputs: Example input tensor for the model
        
        Returns:
            Dictionary mapping Conv2d modules to their channel importance scores
        """
        device = example_inputs.device
        
        # Create a simple data loader from example inputs
        # In practice, this should use actual training data
        batch_size = example_inputs.size(0)

        # Generate synthetic targets for gradient computation
        # Use model's num_classes if available
        num_classes = getattr(self.model, "num_classes", self.num_classes)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        # Create a simple dataset: return per-sample tensors, not whole batch
        class SimpleDataset:
            def __init__(self, inputs, targets):
                self.inputs = inputs
                self.targets = targets
            
            def __len__(self):
                return self.inputs.size(0)
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]

        dataset = SimpleDataset(example_inputs, targets)
        data_loader = DataLoader(dataset, batch_size=min(batch_size, self.num_samples), shuffle=False)
        
        # Compute FGP importance scores
        try:
            importance_scores = compute_fgp_importance(
                self.model, 
                data_loader, 
                device, 
                num_samples=min(self.num_samples, batch_size)
            )
        except Exception as e:
            print(f"FGP importance computation failed: {e}")
            # Fallback: do not prune if importance cannot be computed
            importance_scores = {}
        
        return importance_scores
    
    def set_training_data(self, data_loader: DataLoader):
        """
        Set training data loader for more accurate FGP analysis
        
        Args:
            data_loader: Training data loader
        """
        self.training_loader = data_loader
    
    def importance_scores_with_data(self, data_loader: DataLoader) -> Dict[nn.Module, torch.Tensor]:
        """
        Compute FGP importance scores using actual training data
        
        Args:
            data_loader: Training data loader
        
        Returns:
            Dictionary mapping Conv2d modules to their channel importance scores
        """
        device = next(self.model.parameters()).device
        
        try:
            importance_scores = compute_fgp_importance(
                self.model, 
                data_loader, 
                device, 
                num_samples=self.num_samples
            )
        except Exception as e:
            print(f"FGP importance computation failed: {e}")
            # Fallback to uniform scores
            importance_scores = {}
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d) and module.out_channels > 1:
                    importance_scores[module] = torch.ones(module.out_channels, device=device)
        
        return importance_scores
