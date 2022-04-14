__all__ = ['LoadWeights', 'LoadOptimizerLR', 'LoadOptimizerState']

import logging
import os.path
import re
from typing import Optional

import torch
from flexlearn.context import Context
from flexutils.io.file_system import simplify_path
from flexutils.io.network import download
from flexutils.misc.string import number_ordinal

from utils.adapters import Adapter
from utils.builders.resume import get_checkpoint_path


class LoadWeights(Adapter):

  def __init__(self,
               path: str,
               logger: Optional[logging.Logger] = None,
               **kwargs):
    self._path = path
    self._logger = logger or Context.get_default_argument("logger")
    self._kwargs = kwargs

  def apply(self, model, **kwargs):
    with open(self._path, 'rb') as reader:
      if self._logger is not None:
        self._logger.info(
            f"Loading a model from \"{simplify_path(self._path)}\"...")

      state_dict = torch.load(reader, map_location=torch.device('cpu'))
      if self._kwargs.get("is_torchvision_densenet", False):
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\."
            r"((?:[12])\.(?:weight|bias|running_mean|running_var))$")
        for key in list(state_dict.keys()):
          res = pattern.match(key)
          if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
      model.load_state_dict(state_dict)
    return model

  @classmethod
  def from_checkpoint(cls,
                      root_path: str,
                      min_epoch: Optional[int] = None,
                      max_epoch: Optional[int] = None,
                      **kwargs):
    path = get_checkpoint_path(root_path, min_epoch, max_epoch)
    path = os.path.join(path, "model.pth")
    return cls(path=path, **kwargs)

  @classmethod
  def from_url(cls, url: str, cache_path: str, **kwargs):
    filepath = download(url=url, path=cache_path)
    return cls.from_path(filepath, **kwargs)

  @classmethod
  def from_path(cls, path: str, **kwargs):
    return cls(path=path, **kwargs)


class LoadOptimizerLR(Adapter):

  def __init__(self, path: str, logger: Optional[logging.Logger] = None):
    self._path = path
    self._logger = logger or Context.get_default_argument("logger")

  def apply(self, optimizer: torch.optim.Optimizer, **kwargs):
    with open(self._path, 'rb') as reader:
      if self._logger is not None:
        self._logger.info(f"Loading the learning rate(s) of the optimizer"
                          f" from \"{simplify_path(self._path)}\"...")
      state = torch.load(reader, map_location=torch.device('cpu'))
      if len(optimizer.param_groups) == len(state['param_groups']):
        for index, (group, group_state) in enumerate(
            zip(optimizer.param_groups, state['param_groups'])):
          group['lr'] = group_state['lr']
          self._logger.info(f"The learning rate of {number_ordinal(index)} "
                            f"parameter group: {group['lr']}")
      else:
        raise RuntimeError(
            f"Cannot load optimizer since we have different parameter groups "
            f"(current: {len(optimizer.param_groups)},"
            f" state: {len(state['param_groups'])})")
    return optimizer

  @classmethod
  def from_checkpoint(cls,
                      root_path: str,
                      min_epoch: Optional[int] = None,
                      max_epoch: Optional[int] = None):
    path = get_checkpoint_path(root_path, min_epoch, max_epoch)
    path = os.path.join(path, "optimizer.pth")
    return cls(path=path)


class LoadOptimizerState(LoadOptimizerLR):

  def apply(self, optimizer: torch.optim.Optimizer, **kwargs):
    with open(self._path, 'rb') as reader:
      if self._logger is not None:
        self._logger.info(f"Loading the state of the optimizer"
                          f" from \"{simplify_path(self._path)}\"...")
      state = torch.load(reader, map_location=torch.device('cpu'))
      optimizer.load_state_dict(state)
    return optimizer
