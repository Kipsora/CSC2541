__all__ = ['LogOnce']

import logging


class LogOnce(logging.Filter):
  def __init__(self, levels):
    super().__init__()
    self._levels = {level: set() for level in levels}

  def filter(self, record: logging.LogRecord) -> bool:
    storage = self._levels.get(record.levelno)
    if storage is not None:
      message = record.getMessage()
      if message not in storage:
        storage.add(message)
        return True
      return False
    return True
