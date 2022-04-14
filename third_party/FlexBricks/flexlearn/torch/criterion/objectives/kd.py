__all__ = ['KDLoss']

import torch
from torch.nn import functional


class KDLoss(torch.nn.Module):

  def __init__(self, temperature):
    super().__init__()
    self._temperature = temperature
    self.reduction = 'mean'

  def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
    outputs = functional.log_softmax(outputs / self._temperature, dim=1)
    targets = functional.softmax(targets / self._temperature, dim=1)
    loss = functional.kl_div(outputs, targets, reduction="batchmean")
    loss *= self._temperature * self._temperature
    return loss
