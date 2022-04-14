__all__ = ['WidenLinearLayer']

import torch

from utils.adapters import Adapter


class WidenLinearLayer(Adapter):

  def __init__(self, index: str, new_features):
    self._index = index.split('.')
    self._new_features = new_features

  def apply(self, model, **kwargs):
    original_module = model
    for name in self._index:
      model = getattr(model, name)
    if not isinstance(model, torch.nn.Linear):
      raise ValueError(f"The index \"{'.'.join(self._index)}\" does not refer "
                       f"to a torch.nn.Linear layer")

    with torch.no_grad():
      old_out_features = model.out_features
      old_weight = model.weight.data.clone()
      old_bias = model.bias.data.clone() if model.bias is not None else None

      model.out_features += self._new_features
      new_weight = torch.empty(model.out_features,
                               model.in_features,
                               dtype=model.weight.dtype,
                               device=model.weight.device)
      model.weight = torch.nn.Parameter(new_weight)
      if model.bias is not None:
        new_bias = torch.empty(model.out_features,
                               dtype=model.bias.dtype,
                               device=model.bias.device)
        model.bias = torch.nn.Parameter(new_bias)

      model.reset_parameters()

      model.weight.data[:old_out_features, :] = old_weight
      if model.bias is not None:
        model.bias.data[:old_out_features] = old_bias

    return original_module
