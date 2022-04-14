__all__ = ['EvaluateDatasets']

from typing import Dict, Optional, Union, Collection, Iterable, TYPE_CHECKING

from flexlearn.engine.callbacks import Callback

Inferer = Trainer = None
if TYPE_CHECKING:
  from flexlearn.engine.inferer import Inferer
  from flexlearn.engine.trainer import Trainer


class EvaluateDatasets(Callback):

  def __init__(self,
               inferer: Inferer,
               data_loaders: Dict[str, Iterable],
               plugins: Optional[Union[str, Collection[str]]] = None,
               *,
               use_zeroth_epoch: bool = False):
    if isinstance(plugins, str):
      plugins = {plugins}
    elif plugins is not None:
      plugins = set(plugins)
    else:
      plugins = set()
    self._plugins = plugins
    self._inferer = inferer
    self._data_loaders = data_loaders
    self._use_zeroth_epoch = use_zeroth_epoch

  def _infer(self, **kwargs):
    for dataset, data_loader in self._data_loaders.items():
      if "dataset" in self._plugins:
        kwargs["dataset"] = dataset
      self._inferer.run(data_loader, **kwargs)

  def prior_all(self, engine: Trainer, **kwargs):
    epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    if self._use_zeroth_epoch and epoch_index == 0:
      kwargs = {
          k: self._get_attribute(k, engine, **kwargs) for k in self._plugins
      }
      self._infer(**kwargs)

  def after_epoch(self, engine: Trainer, **kwargs):
    kwargs = {
        k: self._get_attribute(k, engine, **kwargs) for k in self._plugins
    }
    self._infer(**kwargs)
