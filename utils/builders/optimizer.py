__all__ = ["build_optimizer"]

import torch.optim

from utils.adapters.load import LoadOptimizerLR


def apply_adapter(config, model):
  if config.type == 'LoadOptimizerLR':
    if hasattr(config, 'from_checkpoint'):
      adapter = LoadOptimizerLR.from_checkpoint(**config.from_checkpoint)
    else:
      raise RuntimeError("Cannot construct the LoadWeight adapter")
  else:
    raise ValueError(f"Unexpected adapter {config.type}")
  return adapter.apply(model)


def build_optimizer(config, params, resume: bool = False):
  if not hasattr(config, 'kwargs'):
    kwargs = dict()
  else:
    kwargs = dict(config.kwargs)
  kwargs = {k: v for k, v in kwargs.items() if v is not None}

  params = filter(lambda param: param.requires_grad, params)
  if config.type == "SGD":
    optimizer = torch.optim.SGD(params, **kwargs)
  elif config.type == "Adam":
    optimizer = torch.optim.Adam(params, **kwargs)
  else:
    raise ValueError(f"Unsupported optimizer {config.type}")

  if hasattr(config, "adapters"):
    for adapter in config.adapters:
      if not resume or not getattr(adapter, "skip_resumption", False):
        optimizer = apply_adapter(adapter, optimizer)
  return optimizer
