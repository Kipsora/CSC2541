__all__ = ['build_dataset']

import functools
import logging
import math
import os.path
from typing import Optional

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from flexlearn.context import Context
from flexlearn.torch.data_loaders import *
from flexlearn.torch.datasets.image_folders import ImageFolders
from flexlearn.torch.distributed import TorchProcessGroup
from flexutils.io import serialize
from flexutils.io.file_system import simplify_path
from flexutils.misc import AttrDict, NestedDictifier
from utils.datasets import ExtendedImageFolders
from torchvision import transforms as vt


def build_transform(config, logger):
  if not hasattr(config, "kwargs"):
    kwargs = AttrDict()
  else:
    kwargs = AttrDict(config.kwargs)

  if hasattr(kwargs, "interpolation"):
    kwargs.interpolation = getattr(vt.InterpolationMode,
                                   config.interpolation.upper())
  if config.type == "Resize":
    return vt.Resize(**kwargs)
  elif config.type == "RandomResizedCrop":
    return vt.RandomResizedCrop(**kwargs)
  elif config.type == "RandomHorizontalFlip":
    return vt.RandomHorizontalFlip(**kwargs)
  elif config.type == "RandomCrop":
    return vt.RandomCrop(**kwargs)
  elif config.type == "RandomAdjustSharpness":
    return vt.RandomAdjustSharpness(**kwargs)
  elif config.type == "RandomAutocontrast":
    return vt.RandomAutocontrast(**kwargs)
  elif config.type == "RandomEqualize":
    return vt.RandomEqualize(**kwargs)
  elif config.type == "RandomGrayscale":
    return vt.RandomGrayscale(**kwargs)
  elif config.type == "RandomPerspective":
    return vt.RandomPerspective(**kwargs)
  elif config.type == "ToTensor":
    return vt.ToTensor()
  elif config.type == "CenterCrop":
    return vt.CenterCrop(**kwargs)
  elif config.type == "Normalize":
    return vt.Normalize(**kwargs)

  try:
    from torchaudio import transforms as at
    if config.type == "MFCC":
      return at.MFCC(**kwargs)
    elif config.type == "MelScale":
      return at.MelScale(**kwargs)
    elif config.type == "MelSpectrogram":
      return at.MelSpectrogram(**kwargs)
    elif config.type == "AmplitudeToDB":
      return at.AmplitudeToDB(**kwargs)
    elif config.type == "LFCC":
      return at.LFCC(**kwargs)
    elif config.type == "TimeStretch":
      return at.TimeStretch(**kwargs)
  except ModuleNotFoundError:
    if getattr(build_transform, '__warned_torchaudio__', False):
      setattr(build_transform, '__warned_torchaudio__', True)
      logger.warning("torchaudio is not installed and audio transforms "
                     "will not be available.")
  raise ValueError(f"Unsupported transform {config.type}")


def build_transforms(config, logger):
  if len(config) > 1:
    return vt.Compose([build_transform(e, logger) for e in config])
  return build_transform(config[0], logger)


def get_dataset_info(dataset_path, config):
  folders = list()
  ratios = list()
  for partition in config.partitions:
    folders.append(os.path.join(dataset_path, partition.path))
    ratios.append(partition.ratio)
  return folders, ratios


def build_sample_mapping(dataset, *, ratios, shuffle=False):
  if ratios is None:
    return None

  indices = [[list() for _ in dataset.classes] for _ in dataset.folders]
  for index, (_, class_index, path_index) in enumerate(dataset.raw_samples):
    indices[path_index][class_index].append(index)

  if shuffle:
    for path_index in range(len(indices)):
      for class_index in range(len(indices[path_index])):
        np.random.shuffle(indices[path_index][class_index])

  mapping = list()
  ratios = list(ratios)
  for path_index in range(len(indices)):
    for class_index in range(len(indices[path_index])):
      if ratios[path_index] is not None:
        size = len(indices[path_index][class_index]) * ratios[path_index]
        mapping.extend(indices[path_index][class_index][:math.ceil(size)])
      else:
        mapping.extend(indices[path_index][class_index])
  return mapping


def build_dataset(config,
                  data_loader_config,
                  transform_config,
                  dataset_path,
                  num_workers,
                  use_cuda: bool = False,
                  logger: Optional[logging.Logger] = None,
                  training_dataset=None,
                  **kwargs):
  logger = logger or Context.get_default_argument("logger")

  shuffle = kwargs.pop("shuffle", training_dataset is None)

  if hasattr(config, "kwargs"):
    kwargs.update(config.kwargs)
  kwargs = NestedDictifier().dictify(kwargs)

  if hasattr(kwargs, "path"):
    kwargs.path = os.path.join(dataset_path, kwargs.path)
  if hasattr(kwargs, "paths"):
    kwargs.paths = [os.path.join(dataset_path, p) for p in kwargs.paths]
  if hasattr(kwargs, "class_indices"):
    if isinstance(kwargs.class_indices, str):
      logger.info(f"Loading class indices from "
                  f"\"{simplify_path(kwargs.class_indices)}\"...")
      kwargs.class_indices = serialize.load_json(kwargs.class_indices)
  elif training_dataset is not None and \
          hasattr(training_dataset, "class_indices"):
    kwargs.class_indices = training_dataset.class_indices

  if hasattr(data_loader_config, "type"):
    logger.warning(
        "The type for the data loader are now a deprecated parameter and "
        "will be ignored. The type will be automatically determined by "
        "use_cuda.")

  should_split_transform = False
  if hasattr(data_loader_config, "kwargs"):
    collate_fn = getattr(data_loader_config.kwargs, "collate_fn", None)
    if collate_fn == "faster_image_collator":
      should_split_transform = True

  if should_split_transform:
    indices = []
    for i in reversed(range(len(transform_config))):
      if transform_config[i].type == "ToTensor":
        indices.append(i)
    if len(indices) != 1:
      raise ValueError("Cannot find the exact ToTensor transform, which is "
                       "required for building CUDADatasetPrefetcher")
    prior_transform = build_transforms(transform_config[:indices[0]], logger)
    after_transform = build_transforms(transform_config[indices[0] + 1:],
                                       logger)

    def batch_transform(batch):
      batch['source'] = after_transform(batch['source'])
      return batch
  else:
    prior_transform = build_transforms(transform_config, logger)
    after_transform = None
    batch_transform = None

  if config.type == "ImageFolders":
    dataset = ImageFolders(**kwargs, transform=prior_transform)
  elif config.type == "ExtendedImageFolders":
    folders, ratios = get_dataset_info(dataset_path, config)
    kwargs.paths = folders
    dataset = ExtendedImageFolders(transform=prior_transform, **kwargs)
    dataset.set_sample_mapping(build_sample_mapping(dataset, ratios=ratios))
  else:
    raise ValueError(f"Unsupported dataset {config.type}")

  data_loader = build_data_loader(data_loader_config,
                                  dataset,
                                  shuffle=shuffle,
                                  use_cuda=use_cuda,
                                  num_workers=num_workers,
                                  batch_transform=batch_transform)
  return AttrDict(dataset=dataset, data_loader=data_loader)


def build_collate_fn(config, transform=None):
  if config == "faster_image_collator":
    return functools.partial(faster_image_collator, transform=transform)
  raise ValueError(f"Unsupported collate function {config}")


def build_data_loader(config,
                      dataset,
                      num_workers,
                      shuffle=False,
                      use_cuda=False,
                      batch_transform=None,
                      process_group: Optional[TorchProcessGroup] = None):
  process_group = process_group or Context.get_default_argument("process_group")
  if not hasattr(config, "kwargs"):
    kwargs = AttrDict()
  else:
    kwargs = AttrDict(config.kwargs)

  sampler = None
  if process_group is not None:
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=process_group.global_size(),
        rank=process_group.global_rank(),
        shuffle=shuffle)
  else:
    kwargs.shuffle = shuffle

  if hasattr(kwargs, "collate_fn"):
    kwargs.collate_fn = build_collate_fn(kwargs.collate_fn,
                                         transform=batch_transform)
  if hasattr(kwargs, "batch_size") and process_group is not None:
    size = process_group.global_size()
    rank = process_group.global_rank()
    kwargs.batch_size = (kwargs.batch_size // size)
    kwargs.batch_size += int(rank < kwargs.batch_size % size)

  kwargs.num_workers = num_workers
  kwargs.persistent_workers = True

  if use_cuda:
    return CUDADatasetPrefetcher(dataset, sampler=sampler, **kwargs)
  return DataLoader(dataset, sampler=sampler, pin_memory=use_cuda, **kwargs)
