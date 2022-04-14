__all__ = ['load_torch', 'dump_torch']

from flexutils.io.file_system import as_file_object
from flexutils.io.serialize import registry

try:
  import torch

  def load_torch(file, **kwargs):
    with as_file_object(file, mode='rb') as reader:
      return torch.load(reader, **kwargs)

  def dump_torch(data, file, **kwargs):
    with as_file_object(file, mode='wb') as writer:
      torch.save(data, writer, **kwargs)

  registry.register('load', '.json', load_torch)
  registry.register('dump', '.json', dump_torch)

  registry.register('load', 'json', load_torch)
  registry.register('dump', 'json', dump_torch)
except ModuleNotFoundError:
  pass
