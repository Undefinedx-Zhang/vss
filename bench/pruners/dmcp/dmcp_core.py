"""
DMCP Core Implementation
Adapted from: https://github.com/zx55/dmcp

Key concepts:
- Markov process for channel pruning decisions
- Differentiable gate variables for each channel
- Two-stage training: normal training + DMCP optimization
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class MarkovGate(nn.Module):
    """Differentiable gate for channel selection using Markov process"""
    
    def __init__(self, num_channels: int, temperature: float = 1.0):
        super().__init__()
        self.num_channels = num_channels
        self.temperature = temperature
        
        # Markov transition matrix parameters (learnable)
        self.trans_weights = nn.Parameter(torch.ones(num_channels, num_channels))
        self.init_probs = nn.Parameter(torch.ones(num_channels))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input tensor [N, C, H, W]
        Returns:
            gated_x: output with channel gates applied
            gate_probs: channel selection probabilities
        """
        batch_size = x.size(0)
        
        # Compute transition probabilities using softmax
        trans_probs = F.softmax(self.trans_weights / self.temperature, dim=1)
        init_probs = F.softmax(self.init_probs / self.temperature, dim=0)
        
        # Sample channel selection using Gumbel-Softmax for differentiability
        if self.training:
            # Gumbel-Softmax sampling for differentiable discrete sampling
            logits = torch.log(init_probs + 1e-8)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
            gates = F.softmax((logits + gumbel_noise) / self.temperature, dim=0)
        else:
            # During inference, use hard selection
            gates = torch.zeros_like(init_probs)
            gates[torch.argmax(init_probs)] = 1.0
        
        # Apply gates to channels
        gates = gates.view(1, -1, 1, 1).expand_as(x)
        gated_x = x * gates
        
        return gated_x, gates.mean(dim=(0, 2, 3))  # return channel-wise gate probabilities


class DMCPConv2d(nn.Module):
    """Conv2d layer with DMCP gate"""
    
    def __init__(self, conv_layer: nn.Conv2d, gate_temperature: float = 1.0):
        super().__init__()
        self.conv = conv_layer
        self.gate = MarkovGate(conv_layer.out_channels, gate_temperature)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(x)
        gated_out, gate_probs = self.gate(conv_out)
        return gated_out, gate_probs


def wrap_model_with_dmcp_gates(model: nn.Module, temperature: float = 1.0) -> Tuple[nn.Module, List[MarkovGate]]:
    """Wrap Conv2d layers in the model with DMCP gates"""
    gates = []
    
    def replace_conv_recursive(module, name=""):
        for child_name, child in module.named_children():
            if isinstance(child, nn.Conv2d) and child.out_channels > 1:
                # Replace Conv2d with DMCPConv2d
                dmcp_conv = DMCPConv2d(child, temperature)
                setattr(module, child_name, dmcp_conv)
                gates.append(dmcp_conv.gate)
            else:
                replace_conv_recursive(child, f"{name}.{child_name}" if name else child_name)
    
    replace_conv_recursive(model)
    return model, gates


def compute_dmcp_loss(gate_probs_list: List[torch.Tensor], target_flops_ratio: float = 0.5) -> torch.Tensor:
    """
    Compute DMCP regularization loss to encourage target FLOPs ratio
    
    Args:
        gate_probs_list: List of gate probabilities from each DMCP layer
        target_flops_ratio: Target ratio of remaining FLOPs (0.5 = 50% pruning)
    """
    if not gate_probs_list:
        return torch.tensor(0.0)
    
    # Approximate FLOPs reduction based on gate probabilities
    total_gates = 0
    active_gates = 0
    
    for probs in gate_probs_list:
        total_gates += probs.numel()
        active_gates += probs.sum()
    
    if total_gates == 0:
        return torch.tensor(0.0)
    
    current_ratio = active_gates / total_gates
    target_ratio = torch.tensor(target_flops_ratio, device=current_ratio.device)
    
    # L2 loss to encourage target ratio
    ratio_loss = F.mse_loss(current_ratio, target_ratio)
    
    # Entropy regularization to encourage diversity in gate selection
    entropy_loss = 0.0
    for probs in gate_probs_list:
        # Add small epsilon to avoid log(0)
        probs_safe = probs + 1e-8
        entropy = -(probs_safe * torch.log(probs_safe)).sum()
        entropy_loss += entropy
    
    entropy_loss = entropy_loss / len(gate_probs_list) if gate_probs_list else 0.0
    
    return ratio_loss + 0.01 * entropy_loss  # Small entropy weight


def extract_channel_config(model: nn.Module, gate_threshold: float = 0.1) -> Dict[str, List[int]]:
    """Extract channel configuration from trained DMCP gates"""
    config = {}
    
    for name, module in model.named_modules():
        if isinstance(module, DMCPConv2d):
            # Get gate probabilities
            with torch.no_grad():
                dummy_input = torch.randn(1, module.conv.in_channels, 32, 32)
                if next(module.parameters()).is_cuda:
                    dummy_input = dummy_input.cuda()
                _, gate_probs = module.gate(dummy_input)
                
                # Select channels above threshold
                selected_channels = (gate_probs > gate_threshold).nonzero(as_tuple=True)[0].tolist()
                config[name] = selected_channels
    
    return config


def apply_channel_config(model: nn.Module, channel_config: Dict[str, List[int]]) -> nn.Module:
    """Apply channel configuration to create pruned model"""
    # This is a simplified version - in practice, you'd need to handle
    # dependencies between layers and update subsequent layers accordingly
    
    for name, selected_channels in channel_config.items():
        module_parts = name.split('.')
        current_module = model
        
        # Navigate to the target module
        for part in module_parts[:-1]:
            current_module = getattr(current_module, part)
        
        dmcp_conv = getattr(current_module, module_parts[-1])
        if isinstance(dmcp_conv, DMCPConv2d):
            # Extract original conv layer
            original_conv = dmcp_conv.conv
            
            # Create new conv layer with selected channels
            if len(selected_channels) > 0:
                new_out_channels = len(selected_channels)
                new_conv = nn.Conv2d(
                    original_conv.in_channels,
                    new_out_channels,
                    original_conv.kernel_size,
                    original_conv.stride,
                    original_conv.padding,
                    original_conv.dilation,
                    original_conv.groups,
                    original_conv.bias is not None
                )
                
                # Copy selected channel weights
                with torch.no_grad():
                    new_conv.weight.data = original_conv.weight.data[selected_channels]
                    if original_conv.bias is not None:
                        new_conv.bias.data = original_conv.bias.data[selected_channels]
                
                # Replace the DMCP conv with regular conv
                setattr(current_module, module_parts[-1], new_conv)
    
    return model
