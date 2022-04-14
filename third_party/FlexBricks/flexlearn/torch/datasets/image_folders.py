__all__ = ['ImageFolders']

from typing import Optional, Dict, Callable

import torchvision
from torch.utils.data import Dataset

from flexlearn.misc.traverse import traverse_image_folders


class ImageFolders(Dataset):

  def __init__(self,
               paths,
               *,
               transform: Optional[Callable] = None,
               class_indices: Optional[Dict[str, int]] = None,
               image_loader: Optional[Callable] = None,
               use_path_index: bool = False,
               use_item_index: bool = False):
    self._transform = transform
    self._image_loader = image_loader
    if self._image_loader is None:
      self._image_loader = torchvision.datasets.folder.default_loader
    self._folders, self._samples, self._classes, self._class_indices = \
      traverse_image_folders(paths, class_indices=class_indices)
    self._use_path_index = use_path_index
    self._use_item_index = use_item_index

  @property
  def folders(self):
    return self._folders

  @property
  def classes(self):
    return self._classes

  @property
  def class_indices(self):
    return self._class_indices

  @property
  def class_to_idx(self):
    return self._class_indices

  @property
  def samples(self):
    return self._samples

  def __getitem__(self, item_index):
    path, label, path_index = self._samples[item_index]
    image = self._image_loader(path)
    if self._transform is not None:
      image = self._transform(image)
    result = dict(source=image, target=label)
    if self._use_path_index:
      result['path_index'] = path_index
    if self._use_item_index:
      result['item_index'] = item_index
    return result

  def __len__(self):
    return len(self._samples)
