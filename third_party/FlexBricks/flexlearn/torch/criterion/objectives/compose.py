__all__ = ['ComposeObjective']

from typing import Dict

import torch

from flexutils.misc import AttrDict


def get_unique_reduction(reduction):
  if not isinstance(reduction, dict):
    return reduction

  if not reduction:
    raise ValueError("Cannot infer reduction from an empty reduction map")

  unique_reduction = None
  for key, value in reduction.items():
    current_reduction = get_unique_reduction(value)
    if unique_reduction is None:
      unique_reduction = current_reduction
    elif unique_reduction != current_reduction:
      raise ValueError(f"Inconsistent reduction is found "
                       f"(\"{unique_reduction}\" != \"{current_reduction})\"")
  return unique_reduction


class ComposeObjective(torch.nn.Module):
  """
  ComposeObjective sums up a collection of objectives into one. The summed
  objective will be stored at the `None` entry of the output dictionary.
  """

  def __init__(self, objectives: Dict[str, torch.nn.Module]):
    super().__init__()
    if not objectives:
      raise ValueError("Cannot compose empty objectives")
    self._objectives = torch.nn.ModuleDict(objectives)

    self.reduction = dict()
    for k, v in self._objectives.items():
      self.reduction[k] = v.reduction
    self.reduction[None] = get_unique_reduction(self.reduction)

  def forward(self, outputs, targets):
    results = AttrDict()
    loss = None
    for k, v in self._objectives.items():
      current_loss = v(outputs, targets)
      if loss is None:
        loss = torch.zeros_like(current_loss)
      loss += current_loss
      results[k] = current_loss
    results[None] = loss
    return results
