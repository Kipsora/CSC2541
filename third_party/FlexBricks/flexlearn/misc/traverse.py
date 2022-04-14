__all__ = ['traverse_image_folders']

import os
from typing import Optional, Collection, Union, Dict


def traverse_image_folders(paths: Union[str, Collection[str]],
                           *,
                           class_indices: Optional[Dict[str, int]] = None):
  paths = [paths] if isinstance(paths, str) else list(paths)
  folders = list()
  samples = list()

  has_class_indices = class_indices is not None
  class_indices = class_indices.copy() if has_class_indices else dict()

  for path_index, path in enumerate(paths):
    folders.append(path)
    for class_name in sorted(os.listdir(path)):
      class_path = os.path.join(path, class_name)
      if not os.path.isdir(class_path):
        continue
      if class_name not in class_indices:
        if not has_class_indices:
          class_indices[class_name] = len(class_indices)
        else:
          raise ValueError(
              f"Cannot found class {class_name} in the given class_indices")

      class_index = class_indices[class_name]

      image_paths = [
          os.path.join(class_path, i) for i in os.listdir(class_path)
      ]
      image_paths = sorted(list(filter(os.path.isfile, image_paths)))
      for image_path in image_paths:
        samples.append((image_path, class_index, path_index))

  class_names = {v: k for k, v in class_indices.items()}
  classes = [class_names[i] for i in range(len(class_indices))]

  return folders, samples, classes, class_indices
