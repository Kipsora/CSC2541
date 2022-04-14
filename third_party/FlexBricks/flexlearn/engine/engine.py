__all__ = ['Engine']

import abc
from typing import Any, Dict, Optional

from flexlearn.engine.callbacks import Callback


class Engine(object, metaclass=abc.ABCMeta):
  """
  :class:`Engine` is the base class for conduct any iterative or repetitive
  workloads. This decouples the core iterative loops from interactions with
  other classes and hence makes it much easier to customize the iterative loop.

  More specifically, there are two way to customize the iterative loop:

  - Implement a subclass that inherits from :class:`Engine`, which is preferred
    if the loop should be fundamentally modified.
  - Implement a :class:`~flexlearn.engine.callbacks.Callback` instance.
    This provides a simpler API that can tweak the loop.

  Args:
    callback: The callback for the engine.
  """

  def __init__(self, callback: Optional[Callback] = None):
    self._callback = callback or Callback()

  def set_callback(self, callback: Callback):
    """
    Set the callback instance for the engine.

    Args:
      callback: The callback for the engine.
    """
    self._callback = callback

  def state_dict(self) -> Dict[str, Any]:
    """
    The current state of the engine.

    Returns:
      A state dict.
    """
    return dict()

  def load_state_dict(self, state: Dict[str, Any], **kwargs):
    """
    Reset the engine's state with a state dict.

    Args:
      state: The state
      **kwargs: Implementation specific arguments.

    Returns:

    """
    pass

  @property
  def callback(self):
    """
    The callback instance of the engine.
    """
    return self._callback

  def _prior_all(self):
    self._callback.prior_all(self)

  def _after_all(self):
    self._callback.after_all(self)

  def _on_exception(self, exception: BaseException):
    self._callback.on_exception(self, exception)
