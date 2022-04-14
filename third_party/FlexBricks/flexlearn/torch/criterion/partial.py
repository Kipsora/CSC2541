__all__ = ['PartialCriterion']

from typing import Optional

import torch.nn

from flexutils.misc import dict_walk


class PartialCriterion(torch.nn.Module):

  def __init__(self,
               criterion: torch.nn.Module,
               output_path: Optional[str] = None,
               target_path: Optional[str] = None):
    super().__init__()
    self._criterion = criterion

    self._output_path = None
    if output_path is not None:
      self._output_path = output_path.split('.')

    self._target_path = None
    if target_path is not None:
      self._target_path = target_path.split('.')

    self.reduction = self._criterion.reduction

  def forward(self, outputs, targets):
    sources = {'output': outputs, 'target': targets}
    if self._output_path is not None:
      outputs = dict_walk(sources, self._output_path)
    if self._target_path is not None:
      targets = dict_walk(sources, self._target_path)
    return self._criterion(outputs, targets)
