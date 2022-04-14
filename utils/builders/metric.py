__all__ = ['build_metric']

from flexlearn.torch.criterion.metrics import *
from flexlearn.torch.criterion.partial import PartialCriterion


def build_metric(config):
  if not hasattr(config, "kwargs"):
    kwargs = dict()
  else:
    kwargs = dict(config.kwargs)
  if config.type == "TopKAccuracy":
    metric = TopKAccuracy(**kwargs)
  elif config.type == 'Compose':
    metrics = dict()
    for key, metric in config.objectives.items():
      metrics[key] = build_metric(metric)
    metric = ComposeMetric(metrics)
  else:
    raise ValueError(f"Unsupported metric {config.type}")
  output_path = getattr(config, 'output_path', None)
  target_path = getattr(config, 'target_path', None)
  if output_path is not None or target_path is not None:
    metric = PartialCriterion(metric, output_path, target_path)
  return metric
