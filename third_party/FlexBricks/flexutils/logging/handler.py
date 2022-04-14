__all__ = ['CallbackHandler']

import logging


class CallbackHandler(logging.Handler):

  def __init__(self, callback, level=logging.NOTSET):
    super().__init__(level)
    assert callable(callback)
    self._callback = callback

  def emit(self, record: logging.LogRecord) -> None:
    try:
      msg = self.format(record)
      self._callback(msg)
      self.flush()
    except (KeyboardInterrupt, SystemExit):
      raise
    except:
      self.handleError(record)
