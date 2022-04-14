__all__ = ['control_reproducibility']

import random

import numpy as np
import torch
import torch.backends.cudnn


def control_reproducibility(seed):
  if seed is None:
    torch.backends.cudnn.benchmark = True
    return

  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.use_deterministic_algorithms(mode=True, warn_only=True)
