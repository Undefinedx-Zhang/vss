from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from ..base import Pruner, PruningConfig


def geometric_median_distance(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute per-filter distance to geometric median approximated by pairwise distances.
    weight: (out_channels, in_channels, kH, kW)
    Returns importance scores where smaller distance => more redundant => lower importance.
    """
    oc = weight.shape[0]
    w = weight.view(oc, -1)
    # pairwise L2 distances
    dists = torch.cdist(w, w, p=2)
    # distance to median: sum to others (proxy)
    scores = dists.sum(dim=1)
    # importance = scores (larger distance => more unique => higher importance)
    return scores


class FPGMPruner(Pruner):
    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        model = self.model
        device = example_inputs.device
        model.eval()

        scores: Dict[nn.Module, torch.Tensor] = {}
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                w = m.weight.detach()
                s = geometric_median_distance(w).to(device)
                scores[m] = s
        return scores


