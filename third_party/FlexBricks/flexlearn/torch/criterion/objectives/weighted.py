__all__ = ['WeightedObjective']

import torch.nn


class WeightedObjective(torch.nn.Module):

  def __init__(self, objective: torch.nn.Module, weight):
    super().__init__()
    self._objective = objective
    self._weight = weight
    self.reduction = self._objective.reduction

  def forward(self, outputs, targets):
    return self._objective(outputs, targets) * self._weight
