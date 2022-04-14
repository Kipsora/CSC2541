__all__ = ['ComposeMetric']

from typing import Dict

import torch


class ComposeMetric(torch.nn.Module):

  def __init__(self, metrics: Dict[str, torch.nn.Module]):
    super().__init__()
    if not metrics:
      raise ValueError("Cannot compose empty metrics")
    self._metrics = torch.nn.ModuleDict(metrics)

    self.reduction = dict()
    for k, v in self._metrics.items():
      self.reduction[k] = v.reduction

  def forward(self, outputs, targets):
    return {k: v(outputs, targets) for k, v in self._metrics.items()}
