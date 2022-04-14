__all__ = ['ManagedCheckpoint']

import logging
import os
from typing import Any, Dict, Optional, List

from flexlearn.checkpoints import Checkpoint
from flexlearn.context import Context
from flexutils.io.file_system import *


class ManagedCheckpoint(Checkpoint):
  """
  :class:`ManagedCheckpoint` provides a convenient API to manages all saved
  checkpoints in terms of steps. An example checkpoint folder will be like: ::

    checkpoints/
      3/
        model.pth
        optimizer.pth
      4/
        model.pth
        optimizer.pth
      7/
        model.pth
        optimizer.pth

  All the subdirectories in ``checkpoints`` should be a number
  (representing the index of the checkpoint) and all non-number named folders
  will not be recognized by :class:`ManagedCheckpoint`.

  .. note::
    The actual layout in the checkpoint is determined by the argument
    ``checkpoint`` and can be different from the example above.

  Args:
    root_path: The root path to the checkpoint folder.
    checkpoint: An :class:`Checkpoint` instance
      to load/save a checkpoint.
    max_checkpoints: Maximum number of checkpoints to be kept. If it is
      zero, then no checkpoint will be removed.
    purge_branch: Whether to purge branched checkpoints. This
      option is useful when the user wants to remove older checkpoints that
      however has larger index to keep the history not branched.
    logger: A logger instance to report errors and infos (if any).
  """

  def __init__(self,
               root_path: str,
               checkpoint: Checkpoint,
               max_checkpoints: int = 20,
               purge_branch: bool = True,
               logger: Optional[logging.Logger] = None):
    self._root_path = get_relative_path(root_path)
    self._checkpoint = checkpoint
    self._max_checkpoints = max_checkpoints
    self._purge_branch = purge_branch
    self._logger = logger or Context.get_default_argument("logger")

  @property
  def metadata(self) -> Dict[str, Any]:
    return self._checkpoint.metadata

  def list_checkpoints(self) -> List[int]:
    """
    List all possible checkpoints

    Returns:
      A list of indices of saved checkpoints.
    """

    saved_checkpoints = []
    for basename in os.listdir(self._root_path):
      if basename.isdigit():
        saved_checkpoints.append(int(basename))
        if not os.path.isdir(os.path.join(self._root_path, basename)):
          self._logger.error(
              f"Found a non-directory checkpoint whose name is a number. "
              f"We treat the file as a incorrect checkpoint and will be "
              f"removed when needed (path="
              f"{os.path.join(self._root_path, basename)}).")
    return saved_checkpoints

  def save(self, index: int, **kwargs):
    """
    Save a checkpoint.

    Args:
      index: The index of the checkpoint to be saved.
      **kwargs: Unused arguments for compatibility.
    """
    if os.path.exists(self._root_path):
      saved_checkpoints = self.list_checkpoints()
      saved_checkpoints = sorted(saved_checkpoints)
      if self._purge_branch:
        while saved_checkpoints[-1] >= index:
          old_index = saved_checkpoints.pop()
          if self._logger is not None:
            self._logger.warning(
                f"Removing the checkpoint with index {old_index} since "
                f"saving current checkpoint {index} can cause branched "
                f"checkpoints...")
          unlink(os.path.join(self._root_path, str(old_index)))

      if self._max_checkpoints > 0:
        saved_checkpoints = list(reversed(saved_checkpoints))
        while len(saved_checkpoints) >= self._max_checkpoints:
          old_index = saved_checkpoints.pop()
          if self._logger is not None:
            self._logger.info(
                f"Removing the checkpoint with index {old_index} for "
                f"keeping only {self._max_checkpoints} checkpoint(s)...")
          unlink(os.path.join(self._root_path, str(old_index)))

    if self._logger is not None:
      self._logger.info(f"Saving the checkpoint with index {index} to "
                        f"{os.path.join(self._root_path, str(index))}...")

    path = os.path.join(self._root_path, str(index))
    self._checkpoint.save(path=path, **kwargs)

  def load(self, index: int, path: Optional[str] = None, **kwargs):
    """
    Load a checkpoint.

    Args:
      index: The index of the checkpoint to be saved.
      path: Optional. The path to the checkpoint to be loaded.
        If not specified, the path will be inferred from the ``index``.
        The ``index`` will be ignored if ``path`` is given.
      **kwargs: Unused arguments for compatibility.
    Returns:
      The loaded checkpoint.
    """
    path = kwargs.get("path")
    if path is None:
      index = kwargs['index']
      kwargs['path'] = os.path.join(self._root_path, str(index))

    self._logger.info(f"Loading a checkpoint from {path}...")
    return self._checkpoint.load(**kwargs)
