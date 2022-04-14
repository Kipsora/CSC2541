__all__ = ['load_numpy', 'dump_numpy']

import numpy as np

from flexutils.io.serialize import registry


def load_numpy(file, **kwargs):
  return np.load(file, **kwargs)


def dump_numpy(file, data, **kwargs):
  return np.save(file, data, **kwargs)


registry.register('load', '.npy', load_numpy)
registry.register('dump', '.npy', dump_numpy)
registry.register('load', 'numpy', load_numpy)
registry.register('dump', 'numpy', dump_numpy)
