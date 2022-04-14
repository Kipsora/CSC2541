__all__ = ['build_objective']

import copy

import torch
from flexlearn.torch.criterion.objectives import *
from flexlearn.torch.criterion.partial import PartialCriterion
from flexutils.misc import AttrDict


def build_objective(config, model):
  if hasattr(config, 'kwargs'):
    kwargs = AttrDict(config.kwargs)
  else:
    kwargs = AttrDict()

  if config.type == "CrossEntropyLoss":
    objective = torch.nn.CrossEntropyLoss(**kwargs)
  elif config.type == "Compose":
    objectives = dict()
    for key, objective in config.objectives.items():
      objectives[key] = build_objective(objective, model)
    objective = ComposeObjective(objectives)
  else:
    raise ValueError(f"Unsupported objective {config.type}")

  output_path = getattr(config, 'output_path', None)
  target_path = getattr(config, 'target_path', None)
  if output_path is not None or target_path is not None:
    objective = PartialCriterion(objective, output_path, target_path)
  weight = getattr(config, 'weight', None)
  if weight is not None:
    objective = WeightedObjective(objective, weight)
  return objective
