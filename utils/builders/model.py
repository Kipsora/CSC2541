__all__ = ["build_model"]

import os

from utils.adapters import *


def apply_adapter(config, model, cache_path):
  if not hasattr(config, "kwargs"):
    kwargs = dict()
  else:
    kwargs = dict(config.kwargs)

  if 'cache_path' in kwargs:
    kwargs['cache_path'] = os.path.join(cache_path, kwargs['cache_path'])

  if config.type == 'LoadWeights':
    if hasattr(config, 'from_checkpoint'):
      adapter = LoadWeights.from_checkpoint(**config.from_checkpoint, **kwargs)
    elif hasattr(config, 'from_url'):
      url = config.from_url.url
      cache_path = os.path.join(cache_path, config.from_url.cache_path)
      adapter = LoadWeights.from_url(url=url, cache_path=cache_path, **kwargs)
    elif hasattr(config, 'from_path'):
      adapter = LoadWeights.from_path(config.from_path, **kwargs)
    else:
      raise RuntimeError("Cannot construct the LoadWeight adapter")
  elif config.type == 'WidenLinearLayer':
    adapter = WidenLinearLayer(**kwargs)
  else:
    raise ValueError(f"Unexpected adapter {config.type}")
  return adapter.apply(model)


def build_model(config, resume: bool = False, *, cache_path: str):
  if not hasattr(config, "kwargs"):
    kwargs = dict()
  else:
    kwargs = dict(config.kwargs)

  if config.arch == "mobilenetv2":
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(**kwargs)
  elif config.arch == "resnet18":
    from torchvision.models import resnet18
    model = resnet18(**kwargs)
  elif config.arch == 'densenet121':
    from torchvision.models import densenet121
    model = densenet121(**kwargs)
  else:
    raise ValueError(f"Unsupported architecture {config.arch}")
  if hasattr(config, "adapters"):
    for adapter in config.adapters:
      if not resume or not getattr(adapter, "skip_resumption", False):
        model = apply_adapter(adapter, model, cache_path)
  return model
