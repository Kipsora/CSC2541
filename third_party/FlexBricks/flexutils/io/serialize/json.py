__all__ = ['load_json', 'loads_json', 'dump_json', 'dumps_json']

import json

from flexutils.io.file_system import as_file_object
from flexutils.io.serialize import registry


class JSONExtendedEncoder(json.JSONEncoder):

  def default(self, o):
    # Adapted from: https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    if hasattr(o, '__jsonify__'):
      return o.__jsonify__()
    if hasattr(o, '__jsonstr__'):
      return o.__jsonstr__

    try:
      import numpy as np
      if isinstance(o, np.integer):
        return int(o)
      if isinstance(o, np.floating):
        return float(o)
      if isinstance(o, (np.ndarray,)):
        return o.tolist()
    except ModuleNotFoundError:
      pass
    try:
      return super(JSONExtendedEncoder, self).default(o)
    except TypeError:
      return str(o)


def load_json(file, **kwargs):
  import json
  with as_file_object(file, mode='r') as reader:
    return json.load(reader, **kwargs)


def dump_json(data, file, **kwargs):
  import json
  kwargs.setdefault('indent', 4)
  kwargs.setdefault('cls', JSONExtendedEncoder)
  with as_file_object(file, mode='w') as writer:
    json.dump(data, writer, **kwargs)


def loads_json(data, **kwargs):
  import json
  return json.loads(data, **kwargs)


def dumps_json(data, **kwargs):
  kwargs.setdefault('cls', JSONExtendedEncoder)
  return json.dumps(data, **kwargs)


registry.register('load', '.json', load_json)
registry.register('dump', '.json', dump_json)

registry.register('load', 'json', load_json)
registry.register('dump', 'json', dump_json)
registry.register('loads', 'json', loads_json)
registry.register('dumps', 'json', dumps_json)
