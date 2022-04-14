__all__ = ['RecordCrossEntropyGaussNewtonRayleighAndGradientAngle']

import copy
import itertools
from typing import Optional

import jax
import numpy as np
import optax
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from flexlearn.context import Context
from flexlearn.engine.callbacks import Callback
from flexlearn.summary import Summary, SummaryItemType
from flexlearn.torch.trainers import SimpleTorchTrainer
from flexutils.misc.string import filtered_join


class RecordCrossEntropyGaussNewtonRayleighAndGradientAngle(Callback):

  def __init__(self,
               data_loader: DataLoader,
               num_batches: int,
               frequency: int,
               prefix: Optional[str] = None,
               summary: Optional[Summary] = None):
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)

    self._data_loader = data_loader
    self._num_batches = num_batches

    self._frequency = frequency
    self._prefix = prefix
    self._old_model = None

  def prior_update(self, engine: SimpleTorchTrainer, **kwargs):
    global_step = self._get_attribute("global_step", engine, **kwargs)
    if self._frequency > 0 and global_step % self._frequency:
      return
    self._old_model = getattr(engine, "_cloned_old_model", None)
    if self._old_model is None:
      self._old_model = copy.deepcopy(engine.model)
      setattr(engine, "_cloned_old_model", self._old_model)

  def after_update(self, engine: SimpleTorchTrainer, **kwargs):
    if hasattr(engine, "_cloned_old_model"):
      delattr(engine, "_cloned_old_model")

  def after_batch(self, engine: SimpleTorchTrainer, **kwargs):
    global_step = self._get_attribute("global_step", engine, **kwargs)

    if self._frequency > 0 and global_step % self._frequency:
      self._old_model = None
      return

    old_params = dict(self._old_model.named_parameters())
    tangents = tuple(param.detach() - old_params[name].detach()
                     for name, param in engine.model.named_parameters())
    self._old_model = None

    with torch.no_grad():
      tangent_norm = sum([torch.sum(tangent**2) for tangent in tangents])
      tangent_norm = torch.sqrt(tangent_norm)
      for tangent in tangents:
        tangent.div_(tangent_norm)

    import functorch
    from functorch.experimental import replace_all_batch_norm_modules_
    fmodel, params, buffers = functorch.make_functional_with_buffers(
        engine.model)
    fmodel = replace_all_batch_norm_modules_(fmodel)

    data_loader = self._data_loader
    if self._num_batches > 0:
      data_loader = itertools.islice(data_loader, self._num_batches)

    components = []
    for inputs in data_loader:
      sources = inputs['source']
      targets = inputs['target']

      for source, target in zip(sources, targets):
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)

        def call_model(*params):
          return fmodel(params, buffers, source)

        output, jvp = functorch.jvp(call_model, params, tangents)
        output = output.detach().cpu().numpy()
        jvp = jvp.detach().cpu().numpy()

        target = F.one_hot(target, output.shape[1])
        target = target.cpu().numpy()

        def objective(output):
          return jax.numpy.mean(optax.softmax_cross_entropy(output, target))

        _, hvp = jax.jvp(jax.grad(objective), (output,), (jvp,))
        components.append(jvp @ hvp.T)

    result = np.mean(components)

    self._summary.add_value(filtered_join("/", [self._prefix, "rayleigh"]),
                            result,
                            index=global_step,
                            item_type=SummaryItemType.PER_GLOBAL_STEP)
