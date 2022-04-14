__all__ = ['ExtendedImageFolders']

from typing import *

import torchvision
from flexlearn.torch.datasets.image_folders import ImageFolders


def process_weights(weights: Optional[Collection[float]], *, expected_length):
  if weights is None:
    return None

  weights = list(weights)
  if len(weights) != expected_length:
    raise ValueError(f"The given weights only have {len(weights)} number(s) "
                     f"which is expected to be {expected_length}")
  for index in range(len(weights)):
    if weights[index] is None:
      weights[index] = 1.0
  return weights


class ExtendedImageFolders(ImageFolders):

  def __init__(
      self,
      paths,
      *,
      transform: Optional[Callable] = None,
      class_indices: Optional[Dict[str, int]] = None,
      use_item_index: bool = False,
      use_path_index: bool = False,
      image_loader: Callable = torchvision.datasets.folder.default_loader):
    super().__init__(paths,
                     transform=transform,
                     class_indices=class_indices,
                     image_loader=image_loader,
                     use_path_index=use_path_index,
                     use_item_index=use_item_index)

    # For getting a partition of the original dataset
    self._sample_mapping = None
    self._mapped_samples = None

  def __len__(self):
    if self._sample_mapping is not None:
      return len(self._sample_mapping)
    return len(self._samples)

  def __getitem__(self, item):
    if self._sample_mapping is not None:
      item = self._sample_mapping[item]

    result = super().__getitem__(item)
    return result

  @property
  def raw_length(self):
    return len(self._samples)

  @property
  def raw_samples(self):
    return self._samples

  @property
  def samples(self):
    if self._mapped_samples is None:
      if self._sample_mapping is not None:
        self._mapped_samples = list()
        for item_index in range(len(self)):
          item_index = self._sample_mapping[item_index]
          self._mapped_samples.append(self._samples[item_index])
      else:
        self._sample_mapping = self._samples
    return self._mapped_samples

  @property
  def sample_mapping(self):
    return self._sample_mapping

  def set_sample_mapping(self, mapping: Optional[Sequence[int]]):
    self._sample_mapping = None
    if mapping is not None:
      self._sample_mapping = tuple(mapping)
    self._mapped_samples = None
