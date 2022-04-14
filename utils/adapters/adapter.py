__all__ = ['Adapter']

import abc


class Adapter(object, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def apply(self, adapted, **kwargs):
    pass
