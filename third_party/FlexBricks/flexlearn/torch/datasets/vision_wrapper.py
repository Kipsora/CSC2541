__all__ = ['VisionWrapper']

from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class VisionWrapper(Dataset):

  def __init__(self, dataset: VisionDataset, use_item_index=False):
    self._dataset = dataset
    self._use_item_index = use_item_index

  def __getitem__(self, item):
    source, target = self._dataset[item]
    result = {'source': source, 'target': target}
    if self._use_item_index:
      result['item_index'] = item
    return result

  def __len__(self):
    return len(self._dataset)

  def __repr__(self):
    return repr(self._dataset)
