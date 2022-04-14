__all__ = ['SimpleTorchTrainer']

from typing import Optional, Callable, Iterable

import torch
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler

from flexlearn.engine.callbacks import Callback
from flexlearn.engine.trainer import Trainer
from flexlearn.torch.helpers import get_default_device
from flexutils.misc import dict_walk


class SimpleTorchTrainer(Trainer):

  def __init__(self,
               model: torch.nn.Module,
               optimizer: Optimizer,
               objective: Callable,
               metric: Optional[Callable] = None,
               source_path: Optional[str] = None,
               target_path: Optional[str] = None,
               device: Optional[torch.device] = None,
               callback: Optional[Callback] = None):
    super().__init__(callback)

    self.model = model
    self.metric = metric
    self.device = device or get_default_device()
    self.objective = objective
    self.optimizer = optimizer

    self._source_path = source_path
    self._target_path = target_path

  def _prior_epoch(self, data_loader: Iterable):
    super()._prior_epoch(data_loader)
    sampler = getattr(data_loader, 'sampler', None)
    if isinstance(sampler, DistributedSampler):
      sampler.set_epoch(self.epoch_index)
    self.model.train()

  def _run_forward(self, sources, targets):
    self._prior_forward(sources, targets)
    outputs = self.model(sources)
    objective = self.objective(outputs, targets)
    self._after_forward(sources, targets)
    return objective, outputs

  def _run_zero_grad(self):
    self.optimizer.zero_grad()

  def _run_backward(self, loss):
    self._prior_backward()
    loss.backward()
    self._after_backward()

  def _run_update(self):
    self._prior_update()
    self.optimizer.step()
    self._after_update()

  def _run_batch(self, inputs, **kwargs):
    if self._source_path is not None:
      sources = dict_walk(inputs, self._source_path.split('.'))
    else:
      sources = inputs['source']
    if self._target_path is not None:
      targets = dict_walk(inputs, self._target_path.split('.'))
    else:
      targets = inputs['target']

    objective, outputs = self._run_forward(sources, targets)

    # The objective might be an instance of `ComposeObjective`, so we should
    # use the `None` entry instead when the objective is a dict.
    loss = objective[None] if isinstance(objective, dict) else objective

    self._run_zero_grad()
    self._run_backward(loss)
    self._run_update()

    results = {'output': outputs, 'objective': objective}
    if self.metric is not None:
      results.setdefault('metric', self.metric(outputs, targets))

    return results
