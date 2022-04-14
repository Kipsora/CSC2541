__all__ = ['SimpleTorchInferer']

from typing import Optional, Callable

import torch

from flexlearn.engine.callbacks import Callback
from flexlearn.engine.inferer import Inferer
from flexlearn.torch.helpers import get_default_device


class SimpleTorchInferer(Inferer):

  def __init__(self,
               model: torch.nn.Module,
               objective: Optional[Callable] = None,
               metric: Optional[Callable] = None,
               device: Optional[torch.device] = None,
               callback: Optional[Callback] = None):
    super().__init__(callback)
    self._device = device or get_default_device()
    self._model = model
    self._objective = objective
    self._metric = metric

  def _run_inference(self, data_loader, **kwargs):
    self._model.eval()
    with torch.no_grad():
      for inputs in data_loader:
        self._prior_batch(inputs, **kwargs)
        outputs = self._run_batch(inputs, **kwargs)
        self._after_batch(inputs, outputs, **kwargs)

  def _run_batch(self, inputs, **kwargs):
    source_key = kwargs.get("source_key", "source")

    sources = inputs[source_key].to(self._device)

    self._prior_forward(**kwargs)
    outputs = self._model(sources)
    self._after_forward(**kwargs)

    results = {'output': outputs}
    if self._objective is not None or self._metric is not None:
      target_key = kwargs.get("target_key", "target")
      targets = inputs[target_key].to(self._device)
      if self._objective is not None:
        results.update({'objective': self._objective(outputs, targets)})
      if self._metric is not None:
        results.update({'metric': self._metric(outputs, targets)})
    return results
