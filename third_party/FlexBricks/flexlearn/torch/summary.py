__all__ = ['TorchSummary']

from typing import Optional

import numpy as np
import torch

from flexlearn.context import Context
from flexlearn.summary import *
from flexlearn.torch.distributed import TorchProcessGroup


class TorchSummaryStorage(SummaryStorage):

  def __init__(self,
               *,
               max_size: Optional[int],
               item_type: SummaryItemType,
               reduction: SummaryReduction,
               process_group: Optional[TorchProcessGroup] = None):
    super().__init__(max_size=max_size,
                     item_type=item_type,
                     reduction=reduction)
    self._has_count_tensor = reduction == SummaryReduction.MEAN or \
                             reduction == SummaryReduction.GLOBAL_MEAN or \
                             reduction == SummaryReduction.LOCAL_MEAN
    self._process_group = process_group or Context.get_default_argument(
        "process_group")
    if self._process_group is not None:
      self._device = self._process_group.device
    else:
      self._device = torch.device('cpu')

  def _record_item(self, item: SummaryItem, value, count, reset: bool):
    if isinstance(value, torch.Tensor):
      value = value.detach().to(self._device)

    if self._has_count_tensor:
      if isinstance(count, torch.Tensor):
        count = count.detach().to(self._device)
      elif count is None:
        count = 1

    if reset:
      item.value.copy_(torch.as_tensor(value))
      if self._has_count_tensor:
        item.count.copy_(torch.as_tensor(count))
    else:
      item.value.add_(value)
      if self._has_count_tensor and count is not None:
        item.count.add_(count)

  def _create_item(self, value, count) -> SummaryItem:
    if isinstance(value, torch.Tensor):
      value = value.clone().detach().to(self._device)
    else:
      value = torch.tensor(value, device=self._device)
    if self._has_count_tensor:
      if count is None:
        count = 1
      if isinstance(count, torch.Tensor):
        count = count.clone().detach().to(self._device)
      else:
        count = torch.tensor(count,
                             dtype=torch.int64,
                             device=self._device,
                             requires_grad=False)
      if self.reduction == SummaryReduction.MEAN:
        value.mul_(count)
    else:
      count = None
    return SummaryItem(value, count)

  def value(self, *, index: Optional[int] = None) -> np.ndarray:
    if index is None:
      index = self._indices[-1]
    item = self._storage[index]

    if self._has_count_tensor:
      result = item.value / item.count
    else:
      result = item.value
    if self._device.type != "cpu":
      result = result.cpu()
    return result.numpy()

  def synchronize(self, *, index: Optional[int] = None, **kwargs):
    if self.reduction is None or \
            self.reduction == SummaryReduction.LOCAL_MEAN or \
            self._process_group is None:
      return

    if index is None:
      index = self._indices[-1]
    item = self._storage[index]

    tensors = [item.value]
    if self._has_count_tensor:
      tensors.append(item.count)

    use_async_op = kwargs.get('use_async_op', False)

    futures = [
        self._process_group.all_reduce_tensor(tensor, use_async_op=use_async_op)
        for tensor in tensors
    ]

    if use_async_op:
      return futures

  def __contains__(self, item):
    return item in self._storage


class TorchSummary(Summary):

  def __init__(self, process_group: Optional[TorchProcessGroup] = None):
    super().__init__()
    self._process_group = process_group or Context.get_default_argument(
        "process_group")

  def create_storage(self, reduction: Optional[SummaryReduction],
                     item_type: SummaryItemType,
                     max_size: Optional[int]) -> SummaryStorage:
    return TorchSummaryStorage(max_size=max_size,
                               reduction=reduction,
                               item_type=item_type,
                               process_group=self._process_group)
