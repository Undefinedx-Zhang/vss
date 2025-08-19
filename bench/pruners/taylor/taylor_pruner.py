from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from ..base import Pruner, PruningConfig


class TaylorBNPruner(Pruner):
    """
    Taylor-FO importance using BN gamma and gradient signal:
    importance(channel) ≈ |gamma_c * dL/d(gamma_c)| aggregated over samples
    Follows the spirit of BN-based Taylor in NVlabs/Taylor_pruning.
    """

    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        model = self.model
        device = example_inputs.device
        model.train()  # need grads

        # collect BN modules following Conv2d (by typical pattern Conv->BN)
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

        # run several mini-batches to accumulate gradients on BN gamma
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        inputs = example_inputs
        if inputs.shape[0] == 1:
            # tile to a few samples to make grads non-degenerate
            inputs = inputs.repeat(self.config.taylor_samples, 1, 1, 1)

        # simple loss: L2 of logits to induce non-zero grads without labels
        outputs = model(inputs)
        loss = (outputs ** 2).mean()
        loss.backward()

        scores: Dict[nn.Module, torch.Tensor] = {}
        for conv, bn in conv_to_bn.items():
            gamma = bn.weight
            if gamma is None or gamma.grad is None:
                s = torch.zeros(conv.out_channels, device=device)
            else:
                s = torch.abs(gamma.detach() * gamma.grad.detach())
                # map BN channels to Conv out-channels one-to-one
                if s.numel() != conv.out_channels:
                    s = torch.zeros(conv.out_channels, device=device)
            scores[conv] = s
        return scores


