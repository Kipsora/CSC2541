__all__ = ['TorchCheckpoint']

import os
from typing import Any, Dict

import torch

from flexlearn.checkpoints import Checkpoint
from flexutils.io import serialize
from flexutils.io.file_system import ensure_directory


class TorchCheckpoint(Checkpoint):

  def __init__(self):
    self._saveables = dict()
    self._metadata = dict()

  def attach(self, key: str, saveable):
    if not hasattr(saveable, "state_dict") or \
            not hasattr(saveable, "load_state_dict"):
      raise ValueError("A saveable object should have state_dict and "
                       "load_state_dict method")
    self._saveables[key] = saveable
    return saveable

  @property
  def metadata(self) -> Dict[str, Any]:
    return self._metadata

  def load(self, **kwargs):
    path = kwargs["path"]
    allowed_keys = kwargs.get("keys")
    if allowed_keys is not None:
      allowed_keys = set(allowed_keys)
    for key, saveable in self._saveables.items():
      if allowed_keys is not None and key not in allowed_keys:
        continue
      with open(os.path.join(path, f'{key}.pth'), 'rb') as reader:
        current_kwargs = dict()
        if "kwargs" in kwargs:
          current_kwargs = kwargs["kwargs"].get(key, dict())
        saveable.load_state_dict(torch.load(reader, **current_kwargs))

    if not kwargs.get('ignore_metadata', False):
      with open(os.path.join(path, f'metadata.pkl'), 'rb') as reader:
        if kwargs.get('force_reload_metadata', False):
          self._metadata.clear()
        self._metadata.update(serialize.load_pickle(reader))

  def save(self, **kwargs):
    path = kwargs["path"]
    ensure_directory(path)
    for key, saveable in self._saveables.items():
      with open(os.path.join(path, f'{key}.pth'), 'wb') as writer:
        torch.save(saveable.state_dict(), writer)
    with open(os.path.join(path, f'metadata.pkl'), 'wb') as writer:
      serialize.dump_pickle(self._metadata, writer)
