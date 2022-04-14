__all__ = ['load_pickle', 'dump_pickle', 'loadb_pickle', 'dumpb_pickle']

from flexutils.io.file_system import as_file_object
from flexutils.io.serialize import registry


def _import_pickle():
  try:
    import cPickle as pickle
  except ModuleNotFoundError:
    import pickle
  return pickle


def load_pickle(file, **kwargs):
  pickle = _import_pickle()
  with as_file_object(file, mode='rb') as reader:
    return pickle.load(reader, **kwargs)


def dump_pickle(data, file, **kwargs):
  pickle = _import_pickle()
  with as_file_object(file, mode='wb') as writer:
    pickle.dump(data, writer, **kwargs)


def loadb_pickle(data, **kwargs):
  pickle = _import_pickle()
  return pickle.loads(data, **kwargs)


def dumpb_pickle(data, **kwargs):
  pickle = _import_pickle()
  return pickle.dumps(data, **kwargs)


registry.register('load', '.pkl', load_pickle)
registry.register('dump', '.pkl', dump_pickle)
registry.register('load', 'pickle', load_pickle)
registry.register('dump', 'pickle', dump_pickle)
registry.register('loadb', 'pickle', loadb_pickle)
registry.register('dumpb', 'pickle', dumpb_pickle)
