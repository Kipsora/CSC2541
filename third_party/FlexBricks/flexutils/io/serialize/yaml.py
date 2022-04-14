__all__ = ['loads_yaml', 'load_yaml', 'dump_yaml', 'dumps_yaml']

from flexutils.io.file_system import as_file_object
from flexutils.io.serialize import registry

try:
  import yaml

  def load_yaml(file, **kwargs):
    kwargs.setdefault('Loader', yaml.FullLoader)
    with as_file_object(file, mode='r') as reader:
      return yaml.load(reader, **kwargs)

  def dump_yaml(data, file, **kwargs):
    with as_file_object(file, mode='w') as writer:
      return yaml.dump(data, writer, **kwargs)

  def loads_yaml(data, **kwargs):
    kwargs.setdefault('Loader', yaml.FullLoader)
    return yaml.load(data, **kwargs)

  def dumps_yaml(data, **kwargs):
    kwargs.setdefault('indent', 4)
    return yaml.dump(data, **kwargs)

  registry.register('load', '.yaml', load_yaml)
  registry.register('dump', '.yaml', dump_yaml)
  registry.register('load', '.yml', load_yaml)
  registry.register('dump', '.yml', dump_yaml)
  registry.register('load', 'yaml', load_yaml)
  registry.register('dump', 'yaml', dump_yaml)
  registry.register('loads', 'yaml', loads_yaml)
  registry.register('dumps', 'yaml', dumps_yaml)
except ModuleNotFoundError:
  pass
