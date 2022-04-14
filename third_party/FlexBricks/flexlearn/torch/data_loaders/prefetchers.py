__all__ = ['CUDADatasetPrefetcher']

from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


class CUDADatasetPrefetcher(DataLoader):
  """
  `Reference <https://github.com/NVIDIA/apex/blob/2ca894da7be755711cbbdf56c74bb7
  904bfd8417/examples/imagenet/main_amp.py#L264-L316>`__
  """

  def __init__(self,
               dataset: Dataset,
               *,
               batch_size: int,
               shuffle: bool = False,
               sampler: Optional[Sampler] = None,
               num_workers: int = 0,
               collate_fn: Optional[Callable] = None,
               cuda_device: Optional[torch.device] = None,
               cuda_stream: Optional[torch.cuda.Stream] = None,
               persistent_workers: bool = False,
               worker_init_fn=None):
    super().__init__(dataset,
                     batch_size=batch_size,
                     shuffle=shuffle,
                     sampler=sampler,
                     pin_memory=True,
                     num_workers=num_workers,
                     collate_fn=collate_fn,
                     persistent_workers=persistent_workers,
                     worker_init_fn=worker_init_fn)

    self._cuda_device = cuda_device or torch.cuda.current_device()
    self._cuda_stream = cuda_stream or torch.cuda.Stream(
        device=self._cuda_device)

  def __iter__(self):
    last_batch = None
    for batch in super(CUDADatasetPrefetcher, self).__iter__():
      with torch.cuda.stream(self._cuda_stream):
        for key in batch:
          batch[key] = batch[key].to(self._cuda_device, non_blocking=True)

      if last_batch is not None:
        yield last_batch

      current_stream = torch.cuda.current_stream(self._cuda_device)
      current_stream.wait_stream(self._cuda_stream)
      last_batch = batch
      for key in batch:
        batch[key].record_stream(current_stream)
    if last_batch is not None:
      yield last_batch
