__all__ = ['build_scheduler']

import torch


def build_scheduler(config, optimizer):
  if not hasattr(config, "kwargs"):
    kwargs = dict()
  else:
    kwargs = dict(config.kwargs)
  if config.type == "CosineAnnealing":
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
  elif config.type == 'None':
    return None
  raise ValueError(f"Unsupported scheduler {config.type}")
