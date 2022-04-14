__all__ = ['log_info', 'log_datasets']

import logging
import os
from typing import Optional, Collection

import torch
from flexlearn.context import Context
from flexlearn.distributed import ProcessGroup
from flexutils.io.file_system import simplify_path
from flexutils.misc.string import format_table


def log_info(args,
             session_path,
             gpus: Collection[torch.device],
             process_group: Optional[ProcessGroup] = None,
             logger: Optional[logging.Logger] = None,
             task: str = "training"):
  logger = logger or Context.get_default_argument("logger", required=True)
  process_group = process_group or Context.get_default_argument("process_group")
  gpus = list(gpus)

  if process_group is None:
    if gpus:
      device_mapping = os.environ.get("CUDA_VISIBLE_DEVICES")
      if device_mapping is not None:
        device_mapping = list(map(int, device_mapping.split(",")))
      else:
        device_mapping = list(range(torch.cuda.device_count()))
      if len(gpus) == 1:
        message = f"The standalone {task} will be executed on gpu " \
                  f"{device_mapping[gpus[0].index]}"
      else:
        message = f"The standalone {task} will be executed on gpus " \
                  f"{[device_mapping[gpu.index] for gpu in gpus]}"
    else:
      message = f"The standalone {task} will executed on cpu"
    logger.info(message)
  else:
    row_rank = [
        "Rank",
        str(process_group.global_rank()),
        str(process_group.local_rank()),
        str(process_group.node_rank())
    ]
    row_size = [
        "Size",
        str(process_group.global_size()),
        str(process_group.local_size()),
        str(process_group.node_size())
    ]
    table = format_table(headers=["", "Global", "Local", "Node"],
                         rows=[row_rank, row_size])
    logger.info(
        f"The training will be conducted in distributed mode:\n{table}\n")

  message = "The paths for current training script:\n"
  if args.resume_path:
    message += f"Resume path: {simplify_path(args.resume_path)}\n"
  else:
    message += f"Config path: {simplify_path(args.config_path)}\n"
  message += f"Dataset path: {simplify_path(args.dataset_path)}\n"
  message += f"Session path: {simplify_path(session_path)}\n"
  logger.info(message)


def log_datasets(config: dict, logger: Optional[logging.Logger] = None):
  logger = logger or Context.get_default_argument("logger", required=True)
  headers = ["", "# Items", "# Batches", "Batch Size"]
  rows = []
  for key, config in config.items():
    rows.append([
        key,
        str(len(config.dataset)),
        str(len(config.data_loader)),
        str(config.data_loader.batch_size)
    ])
  logger.info("Datasets:\n" + format_table(headers, rows) + "\n")
