__all__ = ['get_checkpoint_path']

import datetime
import os.path

from flexutils.io.file_system import *

from utils.builders.common import DATE_FORMAT


def get_checkpoint_path(path: str, min_epoch=None, max_epoch=None):
  if not os.path.exists(path):
    raise RuntimeError(f"The restore path does not exist (path=\"{path}\")")
  if os.path.isfile(path):
    raise RuntimeError(f"The restore path does not refer to a "
                       f"valid folder (path=\"{path}\")")
  current_path = get_relative_path(path)
  if os.path.isdir(os.path.join(current_path, "checkpoints")):
    current_path = os.path.join(current_path, "checkpoints")
  else:
    session_path = os.path.join(current_path, os.path.pardir, os.path.pardir)
    if os.path.isdir(os.path.join(session_path, "checkpoints")):
      return current_path
    if os.path.basename(current_path) != "checkpoints":
      available_session_times = []
      for basename in os.listdir(current_path):
        try:
          available_session_times.append(
              datetime.datetime.strptime(basename, DATE_FORMAT))
        except ValueError:
          pass
      if available_session_times:
        available_session_times = sorted(available_session_times)
        session_time = available_session_times[-1].strftime(DATE_FORMAT)
        current_path = os.path.join(current_path, session_time, "checkpoints")
      else:
        raise RuntimeError(f"Cannot detect the checkpoint path from the "
                           f"restore path (path=\"{path}\")")

  available_epochs = []
  for basename in os.listdir(current_path):
    if basename.isdigit():
      epoch = int(basename)
      if min_epoch is not None and epoch < min_epoch:
        continue
      if max_epoch is not None and epoch > max_epoch:
        continue
      available_epochs.append(int(basename))
  available_epochs = sorted(available_epochs)
  return os.path.join(current_path, str(available_epochs[-1]))
