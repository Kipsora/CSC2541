__all__ = ['AttrDict']


class AttrDict(dict):

  def __getitem__(self, name):
    if name not in self and self.__contains__("__fallback__"):
      return self["__fallback__"]
    return super(AttrDict, self).__getitem__(name)

  def __setitem__(self, key, value):
    return super(AttrDict, self).__setitem__(key, value)

  def __contains__(self, item):
    return super(AttrDict, self).__contains__(item)

  def set_fallback(self, fallback):
    self["__fallback__"] = fallback

  def __getattr__(self, name):
    try:
      return self.__getitem__(name)
    except KeyError as exception:
      raise AttributeError(name) from exception

  __setattr__ = __setitem__
  __call__ = __getitem__
