__all__ = ['Checkpoint']

import abc
from typing import Dict, Any


class Checkpoint(object, metaclass=abc.ABCMeta):
  """
  This is the base class of a checkpoint. Any customized implementation that is
  compatible with FlexBricks should inherit from this class.
  """

  @property
  @abc.abstractmethod
  def metadata(self) -> Dict[str, Any]:
    """
    The metadata of this checkpoint. The checkpoint is responsible for loading
    and saving the metadata.
    """
    pass

  @abc.abstractmethod
  def save(self, **kwargs):
    """
    Save the checkpoint.

    Args:
      **kwargs: The arguments for saving the checkpoint.
    """
    pass

  @abc.abstractmethod
  def load(self, **kwargs):
    """
    Load the checkpoint.

    Args:
      **kwargs: The arguments for loading the checkpoint.

    Returns:
      The loaded checkpoint.
    """
    pass
