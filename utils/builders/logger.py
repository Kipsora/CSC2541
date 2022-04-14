__all__ = ['build_logger']

import logging
import os.path
import sys
from typing import Optional

from flexlearn.context import Context
from flexlearn.torch.distributed import TorchProcessGroup
from flexutils.logging.formatter import ColoredFormatter

from utils.builders.common import DATE_FORMAT


def build_logger(name: str,
                 path: Optional[str] = None,
                 time: Optional[str] = None,
                 level=logging.INFO,
                 use_log_file: bool = True,
                 process_group: Optional[TorchProcessGroup] = None):
  process_group = process_group or Context.get_default_argument("process_group")
  logger = logging.getLogger(name)

  if TorchProcessGroup.is_local_master(process_group):
    formatter = ColoredFormatter(
        fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt=DATE_FORMAT)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  if path is not None and time is not None and use_log_file:
    if process_group is None:
      log_file = f"{name}_{time}.log"
    else:
      log_file = f"{name}_{process_group.global_rank()}_{time}.log"

    log_file = os.path.join(path, log_file)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt=DATE_FORMAT)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

  logger.setLevel(logging.DEBUG)

  logger.propagate = False
  return logger
