__all__ = ["smart_load_state", 'get_default_device']

from typing import Dict, Any
from collections import OrderedDict

import torch
import logging


def get_default_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device('cuda', torch.cuda.current_device())
  else:
    return torch.device('cpu')


def smart_load_state(module: torch.nn.Module,
                     state: Dict[str, Any],
                     logger: logging.Logger,
                     *,
                     use_proxy_keys: bool = True,
                     use_strict_load: bool = True):
  if not use_proxy_keys:
    module.load_state_dict(OrderedDict(state), strict=use_strict_load)
    return

  param_keys = set(module.state_dict().keys())
  model_state = OrderedDict()
  for key, value in state.items():
    proxy_keys = list()
    for param_key in param_keys:
      if (not key.endswith("." + param_key) and
          not param_key.endswith("." + key) and key != param_key):
        continue
      if not proxy_keys:
        proxy_keys = [param_key]
        continue

      new_distance = abs(len(param_key) - len(key))
      old_distance = abs(len(proxy_keys[0]) - len(key))
      if new_distance < old_distance:
        proxy_keys = [param_key]
      elif new_distance == old_distance:
        proxy_keys.append(param_key)

    if not proxy_keys:
      if not use_strict_load:
        logger.warning(f'Cannot find any available proxy parameter for "{key}"')
        continue
      else:
        raise RuntimeError(
            f'Cannot find any available proxy parameter for "{key}"')
    elif len(proxy_keys) > 1:
      raise RuntimeError(f"Ambiguous proxy parameters for '{key}' "
                         f"(Found: {tuple(proxy_keys)})")

    if proxy_keys[0] != key:
      logger.debug(f"Parameter proxy mapping: {key} => {proxy_keys[0]}")
    model_state.setdefault(proxy_keys[0], value)

  module.load_state_dict(model_state, strict=use_strict_load)
