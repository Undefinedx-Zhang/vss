"""
DMCP Pruner Implementation
Integrates DMCP with our unified pruning interface
"""
from __future__ import annotations
import copy
from typing import Dict
import torch
import torch.nn as nn

from ..base import Pruner, PruningConfig
from .dmcp_core import wrap_model_with_dmcp_gates, compute_dmcp_loss, extract_channel_config, apply_channel_config


class DMCPPruner(Pruner):
    """
    DMCP (Differentiable Markov Channel Pruning) implementation
    
    This is a two-stage process:
    1. Wrap model with differentiable gates
    2. Train with DMCP loss to learn channel importance
    3. Extract channel configuration and create pruned model
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        super().__init__(model, config)
        self.temperature = 1.0
        self.target_flops_ratio = 1.0 - config.global_ratio  # Convert pruning ratio to remaining ratio
        self.gate_threshold = 0.1
        
    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        """
        For DMCP, importance is determined by the learned gate probabilities
        This method provides a simplified interface for compatibility
        """
        # Create a copy of the model for DMCP training
        dmcp_model = copy.deepcopy(self.model)
        dmcp_model, gates = wrap_model_with_dmcp_gates(dmcp_model, self.temperature)
        dmcp_model.to(example_inputs.device)
        dmcp_model.train()
        
        # Simplified DMCP training loop (in practice, this would be more extensive)
        optimizer = torch.optim.Adam(dmcp_model.parameters(), lr=0.001)
        
        # Mini training loop to learn gate probabilities
        for step in range(50):  # Simplified - normally would train for many epochs
            optimizer.zero_grad()
            
            # Forward pass through DMCP model
            gate_probs_list = []
            
            def forward_hook(module, input, output):
                if hasattr(module, 'gate'):
                    _, gate_probs = module.gate(output[0] if isinstance(output, tuple) else output)
                    gate_probs_list.append(gate_probs)
            
            # Register hooks to collect gate probabilities
            hooks = []
            for name, module in dmcp_model.named_modules():
                if hasattr(module, 'gate'):
                    hook = module.register_forward_hook(forward_hook)
                    hooks.append(hook)
            
            # Forward pass
            try:
                outputs = dmcp_model(example_inputs)
                
                # Compute DMCP loss
                dmcp_loss = compute_dmcp_loss(gate_probs_list, self.target_flops_ratio)
                
                # Simple classification loss (using random targets for this example)
                if len(outputs.shape) > 1:
                    targets = torch.randint(0, outputs.size(1), (outputs.size(0),), device=outputs.device)
                    ce_loss = nn.CrossEntropyLoss()(outputs, targets)
                    total_loss = ce_loss + 0.1 * dmcp_loss  # Small DMCP loss weight
                else:
                    total_loss = dmcp_loss
                
                total_loss.backward()
                optimizer.step()
                
            except Exception as e:
                print(f"DMCP training step {step} failed: {e}")
                break
            finally:
                # Clean up hooks
                for hook in hooks:
                    hook.remove()
        
        # Extract channel configuration from trained gates
        channel_config = extract_channel_config(dmcp_model, self.gate_threshold)
        
        # Convert to importance scores format for compatibility
        scores: Dict[nn.Module, torch.Tensor] = {}
        
        # Map back to original model modules
        original_modules = list(self.model.modules())
        dmcp_modules = list(dmcp_model.modules())
        
        orig_idx = 0
        for dmcp_module in dmcp_modules:
            if hasattr(dmcp_module, 'gate') and hasattr(dmcp_module, 'conv'):
                # Find corresponding original module
                while orig_idx < len(original_modules):
                    orig_module = original_modules[orig_idx]
                    if isinstance(orig_module, nn.Conv2d) and orig_module.out_channels == dmcp_module.conv.out_channels:
                        # Get gate probabilities as importance scores
                        with torch.no_grad():
                            dummy_input = torch.randn(1, dmcp_module.conv.out_channels, 16, 16, device=example_inputs.device)
                            _, gate_probs = dmcp_module.gate(dummy_input)
                            scores[orig_module] = gate_probs.detach()
                        orig_idx += 1
                        break
                    orig_idx += 1
        
        return scores
    
    def prune(self, example_inputs: torch.Tensor) -> nn.Module:
        """
        Perform DMCP pruning and return pruned model
        """
        # Get importance scores (which runs DMCP training internally)
        scores = self.importance_scores(example_inputs)
        
        # Use the base pruner's logic to apply pruning based on scores
        return super().prune(example_inputs)
