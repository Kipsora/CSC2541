__all__ = ['registry', 'load', 'dump', 'loads', 'dumps', 'loadb', 'dumpb']

from flexutils.io.file_system import get_file_extension
from flexutils.misc import RegistryGroup, CallbackRegistry


class SerializerRegistryGroup(RegistryGroup):
  __base_class__ = CallbackRegistry

  def _default_lookup_fallback(self, key, *args, **kwargs):
    raise ValueError(f"Unsupported file extension \"{key}\"")

  def _default_lookup_format_fallback(self, key, *args, **kwargs):
    raise ValueError(f"Invalid serializer \"{key}\"")

  def dispatch_by_path(self, registry_name, key, *args, **kwargs):
    extension = get_file_extension(key)
    callback = self.lookup(registry_name, extension, True,
                           self._default_lookup_fallback)
    return callback(key, *args, **kwargs)

  def dispatch_by_serializer(self, registry_name, key, *args, **kwargs):
    callback = self.lookup(registry_name, key, True,
                           self._default_lookup_format_fallback)
    return callback(*args, **kwargs)


registry = SerializerRegistryGroup()


def load(file, **kwargs):
  if isinstance(file, str):
    return registry.dispatch_by_path("load", file, **kwargs)
  else:
    serializer = kwargs.pop("serializer")
    return registry.dispatch_by_serializer("load", serializer, file, **kwargs)


def dump(data, file, **kwargs):
  if isinstance(file, str):
    return registry.dispatch_by_path("dump", data, file, **kwargs)
  else:
    serializer = kwargs.pop("serializer")
    return registry.dispatch_by_serializer("dump", serializer, data, file,
                                           **kwargs)


def loads(data: str, serializer, **kwargs):
  if isinstance(serializer, str):
    return registry.dispatch_by_serializer("loads", serializer, data, **kwargs)
  elif callable(serializer):
    return serializer(data, **kwargs)
  else:
    raise ValueError(f"Invalid serializer \"{serializer}\"")


def dumps(data, serializer, **kwargs) -> str:
  if isinstance(serializer, str):
    return registry.dispatch_by_serializer("dumps", serializer, data, **kwargs)
  elif callable(serializer):
    return serializer(data, **kwargs)
  else:
    raise ValueError(f"Invalid serializer \"{serializer}\"")


def loadb(data: bytes, serializer, **kwargs):
  if isinstance(serializer, str):
    return registry.dispatch_by_serializer("loadb", serializer, data, **kwargs)
  elif callable(serializer):
    return serializer(data, **kwargs)
  else:
    raise ValueError(f"Invalid serializer \"{serializer}\"")


def dumpb(data, serializer, **kwargs) -> bytes:
  if isinstance(serializer, str):
    return registry.dispatch_by_serializer("dumpb", serializer, data, **kwargs)
  elif callable(serializer):
    return serializer(data, **kwargs)
  else:
    raise ValueError(f"Invalid serializer \"{serializer}\"")
