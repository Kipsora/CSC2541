__all__ = ['faster_image_collator']

import numpy as np
import torch


def faster_image_collator(batch, transform=None):
  """
  `Reference <https://github.com/NVIDIA/apex/blob/2ca894da7be755711cbbdf56c74bb7
  904bfd8417/examples/imagenet/main_amp.py#L28-L41>`__
  """

  sources = [entry['source'] for entry in batch]
  w = sources[0].size[0]
  h = sources[0].size[1]
  tensors = torch.zeros(
      (len(sources), 3, h, w),
      dtype=torch.uint8)  # .contiguous(memory_format=memory_format)
  for i, image in enumerate(sources):
    array = np.array(image, dtype=np.uint8)
    if array.ndim < 3:
      array = np.expand_dims(array, axis=-1)
    array = np.rollaxis(array, 2)
    tensors[i] += torch.from_numpy(array)

  result = dict()
  for entry in batch:
    for key, value in entry.items():
      if key == 'source':
        continue
      if key not in result:
        result[key] = list()
      result[key].append(value)
  for key, value in result.items():
    result[key] = torch.tensor(value)
  result['source'] = tensors.float().div_(255)
  if transform is not None:
    result = transform(result)

  return result
