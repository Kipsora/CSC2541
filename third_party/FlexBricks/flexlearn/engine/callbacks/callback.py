__all__ = ['Callback']

from typing import TYPE_CHECKING

Engine = None
if TYPE_CHECKING:
  from flexlearn.engine import Engine


class Callback(object):
  """
  This is the base class for all engine callbacks. Since engines are quite
  different (e.g. inferers vs trainers), the callback hooks for different
  engines can be different, which however are all defined in this class. The
  rationale behind this is to make some callbacks reusable across different
  engines. For example, the callback
  :class:`~flexlearn.engine.callbacks.ShowEpochProgress` can be used in both
  inference engines and training engines. It is hence a callback
  implementation's responsibility to warn or error out incompatible engines.

  The builtin callbacks ``all``, ``epoch``, and ``batch``, are designed for
  both inferers and trainers, which can be abstracted as follows:

  .. code-block:: python

     # For both inferers and trainers
     try:
       # prior_all is called
       for epoch in range(num_epochs):
         # prior_epoch is called
         for batch in data_loader:
           # prior_batch is called
           # run training or inference on the batch
           # after_batch is called
         # after_epoch is called
     except Exception as exception:
       # on_exception is called
     finally:
       # after_all is called

  For the other callbacks, please refer to the corresponding engines.

  .. note::
    While currently we only have six builtin callbacks, namely, ``all``,
    ``epoch``, ``batch``, ``forward``, ``backward``, and ``update``, with a
    special ``on_exception`` callback, it is free to add other types of
    callbacks. The only caveat is that any new type of callbacks should be
    called somewhere in the engine and
    :class:`~flexlearn.engine.callbacks.ComposeCallback` should be inherited
    and customized if needed.

  .. warning::
    All the methods in this class should be only called from an
    :class:`~flexlearn.engine.Engine` or inside the callback instance.
  """

  def prior_all(self, engine: Engine, **kwargs):
    pass

  def after_all(self, engine: Engine, **kwargs):
    pass

  def prior_epoch(self, engine: Engine, **kwargs):
    pass

  def after_epoch(self, engine: Engine, **kwargs):
    pass

  def prior_batch(self, engine: Engine, **kwargs):
    pass

  def after_batch(self, engine: Engine, **kwargs):
    pass

  def prior_forward(self, engine: Engine, **kwargs):
    pass

  def after_forward(self, engine: Engine, **kwargs):
    pass

  def prior_backward(self, engine: Engine, **kwargs):
    pass

  def after_backward(self, engine: Engine, **kwargs):
    pass

  def prior_update(self, engine: Engine, **kwargs):
    pass

  def after_update(self, engine: Engine, **kwargs):
    pass

  def on_exception(self, engine: Engine, exception: BaseException, **kwargs):
    pass

  @staticmethod
  def _get_attribute(key: str, engine: Engine, **kwargs):
    if key in kwargs:
      return kwargs.get(key)
    return getattr(engine, key, None)
