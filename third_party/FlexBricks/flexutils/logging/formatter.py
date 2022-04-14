__all__ = ['LogColorScheme', 'ColoredFormatter']

import copy
import logging
from typing import Optional

from flexutils.cli.colored import CLIColorFormat


class LogColorScheme(object):

  def __init__(self,
               critical: Optional[CLIColorFormat] = None,
               error: Optional[CLIColorFormat] = None,
               warning: Optional[CLIColorFormat] = None,
               info: Optional[CLIColorFormat] = None,
               debug: Optional[CLIColorFormat] = None,
               time: Optional[CLIColorFormat] = None,
               name: Optional[CLIColorFormat] = None,
               process: Optional[CLIColorFormat] = None,
               thread: Optional[CLIColorFormat] = None,
               lineno: Optional[CLIColorFormat] = None,
               funcname: Optional[CLIColorFormat] = None,
               filename: Optional[CLIColorFormat] = None,
               pathname: Optional[CLIColorFormat] = None):
    self.critical = critical or CLIColorFormat('white', 'on_red')
    self.error = error or CLIColorFormat('red')
    self.warning = warning or CLIColorFormat('yellow')
    self.info = info or CLIColorFormat('blue')
    self.debug = debug or CLIColorFormat('grey')
    self.time = time or CLIColorFormat('green')
    self.name = name or CLIColorFormat('magenta')
    self.process = process or CLIColorFormat('blue')
    self.thread = thread or CLIColorFormat('yellow')
    self.lineno = lineno or CLIColorFormat('yellow', underline=True)
    self.funcname = funcname or CLIColorFormat('cyan')
    self.filename = filename or CLIColorFormat()
    self.pathname = pathname or CLIColorFormat('white', underline=True)


class ColoredFormatter(logging.Formatter):

  def __init__(self,
               fmt=None,
               datefmt=None,
               style='%',
               color_scheme: LogColorScheme = None):
    super().__init__(fmt, datefmt, style)
    self._scheme = color_scheme or LogColorScheme()

  def format(self, record: logging.LogRecord) -> str:
    record = copy.copy(record)
    levelname_color = getattr(self._scheme, record.levelname.lower())
    levelno_color = getattr(self._scheme,
                            logging.getLevelName(record.levelno).lower())

    record.levelname = levelname_color.colored(record.levelname)
    record.levelno = levelno_color.colored(record.levelno)
    record.name = self._scheme.name.colored(record.name)
    record.process = self._scheme.process.colored(record.process)
    record.processName = self._scheme.process.colored(record.processName)
    record.thread = self._scheme.thread.colored(record.thread)
    record.threadName = self._scheme.thread.colored(record.threadName)
    record.lineno = self._scheme.lineno.colored(record.lineno)
    record.pathname = self._scheme.pathname.colored(record.pathname)
    record.funcName = self._scheme.funcname.colored(record.funcName)
    record.filename = self._scheme.filename.colored(record.filename)
    return super(ColoredFormatter, self).format(record)

  def formatTime(self, record: logging.LogRecord, datefmt=...) -> str:
    return self._scheme.time.colored(super().formatTime(record, datefmt))
