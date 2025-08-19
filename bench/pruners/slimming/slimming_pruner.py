from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from ..base import Pruner, PruningConfig


class SlimmingPruner(Pruner):
    """
    Network Slimming: use BN gamma magnitude as importance; either a fixed threshold or global ratio.
    Training stage should add L1 regularization on BN gamma; here we only perform pruning.
    """

    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        model = self.model
        device = example_inputs.device
        model.eval()

        conv_to_bn = {}
        prev = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                prev = m
            elif isinstance(m, nn.BatchNorm2d) and prev is not None:
                conv_to_bn[prev] = m
                prev = None
            else:
                prev = None if not isinstance(m, nn.Conv2d) else prev

        scores: Dict[nn.Module, torch.Tensor] = {}
        for conv, bn in conv_to_bn.items():
            gamma = bn.weight
            if gamma is None:
                s = torch.zeros(conv.out_channels, device=device)
            else:
                s = torch.abs(gamma.detach())
                if s.numel() != conv.out_channels:
                    s = torch.zeros(conv.out_channels, device=device)
            scores[conv] = s
        return scores


