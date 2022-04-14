__all__ = ['build_gpus']

import torch
from flexlearn.torch.distributed import TorchProcessGroup


def build_gpus(use_gpus: bool, distribute: bool):
  if not use_gpus:
    return []
  if torch.cuda.is_available() and distribute:
    current_device = TorchProcessGroup.env_local_rank()
    if current_device is None:
      raise RuntimeError("Cannot deduce the default cuda device "
                         "since local_rank is not set")
    current_device = int(current_device)
    if current_device >= torch.cuda.device_count():
      raise RuntimeError(
          f"Cannot deduce the default cuda device since local_rank "
          f"({current_device}) is more than available gpus "
          f"({torch.cuda.device_count()})")
    return [torch.device("cuda", current_device)]
  return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
