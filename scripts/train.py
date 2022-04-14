import argparse
import datetime
import functools
import os.path
from typing import Optional

import torch

from flexlearn.checkpoints.managed import ManagedCheckpoint
from flexlearn.context import Context
from flexlearn.engine.callbacks import *
from flexlearn.torch.callbacks import *
from flexlearn.torch.checkpoints import TorchCheckpoint
from flexlearn.torch.distributed import TorchProcessGroup
from flexlearn.torch.inferers import SimpleTorchInferer
from flexlearn.torch.meters import TorchGPUTimeMeter, TorchCPUTimeMeter
from flexlearn.torch.summary import TorchSummary
from flexlearn.torch.trainers import SimpleTorchTrainer
from flexutils.io import serialize
from flexutils.io.file_system import *
from flexutils.misc.string import *
from utils.builders.common import DATE_FORMAT
from utils.builders.config import *
from utils.builders.dataset import *
from utils.builders.gpus import build_gpus
from utils.builders.logger import build_logger
from utils.builders.metric import build_metric
from utils.builders.model import build_model
from utils.builders.objective import build_objective
from utils.builders.optimizer import build_optimizer
from utils.builders.resume import get_checkpoint_path
from utils.builders.scheduler import build_scheduler
from utils.helpers import control_reproducibility
from utils.logging import *


def set_default_checkpoint(checkpoint, session_path, max_checkpoints):
  checkpoint_path = os.path.join(session_path, "checkpoints")
  checkpoint = ManagedCheckpoint(checkpoint_path,
                                 checkpoint,
                                 max_checkpoints=max_checkpoints)
  Context.set_default_argument("checkpoint", checkpoint)


def build_time_meters(extra_scopes, use_cuda: bool):
  meters = dict(epoch=TorchGPUTimeMeter() if use_cuda else TorchCPUTimeMeter())
  for scope in set(extra_scopes):
    meters[scope] = TorchGPUTimeMeter() if use_cuda else TorchCPUTimeMeter()
  return meters


def main(args):
  control_reproducibility(args.seed)
  with Context().as_default(), TorchProcessGroup.activate(
      backend=args.backend) as process_group:
    process_group: Optional[TorchProcessGroup]

    Context.set_default_argument("process_group", process_group)

    session_time = datetime.datetime.now().strftime(DATE_FORMAT)
    if process_group is not None:
      session_time = process_group.broadcast(session_time, source=0)
      args.num_workers //= process_group.local_size()

    if args.resume_path is not None:
      args.resume_path = get_checkpoint_path(args.resume_path)
      session_path = get_relative_path(
          os.path.join(args.resume_path, os.path.pardir, os.path.pardir))
      config = build_config_from_session_path(session_path)
    else:
      config = build_config(args.config_path)
      session_path = os.path.join(args.session_path, config.path, session_time)
      if TorchProcessGroup.is_local_master(process_group):
        ensure_directory(session_path)
        serialize.dump_json(config, os.path.join(session_path, "config.json"))
      if process_group is not None:
        process_group.barrier()

    logger = build_logger("train", session_path, session_time)
    Context.set_default_argument("logger", logger)

    log_info(args, session_path, args.gpus)

    checkpoint = TorchCheckpoint()
    set_default_checkpoint(checkpoint, session_path,
                           getattr(config.checkpoint, "max_checkpoints", 20))

    Context.set_default_argument("summary", TorchSummary())

    # Build datasets and data loaders
    config.datasets.training = build_dataset(
        config.datasets.training,
        config.data_loaders.training,
        config.datasets.transforms.training,
        args.dataset_path,
        num_workers=args.num_workers,
        use_cuda=bool(args.gpus))

    config.datasets.validation = build_dataset(
        config.datasets.validation,
        config.data_loaders.inference,
        config.datasets.transforms.inference,
        args.dataset_path,
        use_cuda=bool(args.gpus),
        num_workers=args.num_workers,
        training_dataset=config.datasets.training.dataset)

    if not hasattr(config.datasets, "evaluations"):
      config.datasets.evaluations = dict()

    for k, dataset in config.datasets.evaluations.items():
      config.datasets.evaluations[k] = build_dataset(
          dataset,
          config.data_loaders.inference,
          config.datasets.transforms.inference,
          args.dataset_path,
          use_cuda=bool(args.gpus),
          num_workers=args.num_workers,
          training_dataset=config.datasets.training.dataset)

    # yapf: disable
    log_datasets({
      'Training': config.datasets.training,
      'Validation': config.datasets.validation,
      **{
        f'Evaluation/{k.capitalize()}': v
        for k, v in config.datasets.evaluations.items()
      }
    })
    # yapf: enable

    # Build model, optimizer, scheduler, objective, and metric
    model = build_model(config.model,
                        resume=args.resume_path is not None,
                        cache_path=args.cache_path)
    checkpoint.attach("model", model)
    logger.info(f"Model architecture:\n{model}")
    if process_group is not None:
      model.to(args.default_device)
      model = process_group.parallelize(model)
    elif args.gpus:
      from torch.nn import DataParallel
      model.to(args.default_device)
      if len(args.gpus) > 1:
        model = DataParallel(model, device_ids=args.gpus)

    optimizer = build_optimizer(config.optimizer,
                                model.parameters(),
                                resume=args.resume_path is not None)
    scheduler = build_scheduler(config.scheduler, optimizer)
    objective = build_objective(config.objective, model)
    metric = build_metric(config.metric)

    # Build a inferer
    inferer = SimpleTorchInferer(
        model,
        objective=objective,
        metric=metric,
        callback=ComposeCallback([
            RecordBatchOutputs(prefix="inferer/{dataset}",
                               plugins="dataset",
                               reduction_map={
                                   'objective': objective.reduction,
                                   'metric': metric.reduction
                               },
                               use_suffix_matching=True),
            ShowEpochProgress(header="Epoch {epoch_index}/{num_epochs}",
                              plugins=["epoch_index", "num_epochs", "dataset"],
                              field_formatter=RegularReplacer({
                                  r"inferer/{dataset}/metric/(.*)": r"\1",
                                  r"inferer/{dataset}/objective": r"loss"
                              }),
                              max_field_length=6),
        ]),
        device=args.default_device)

    # Build a trainer
    inferer_data_loaders = {
        f'evaluation/{key}': value.data_loader
        for key, value in config.datasets.evaluations.items()
    }
    inferer_data_loaders['validation'] = config.datasets.validation.data_loader

    time_meters = build_time_meters(args.meter, use_cuda=bool(args.gpus))

    trainer_callbacks = [
        RecordBatchOutputs(prefix="trainer",
                           reduction_map={
                               'objective': objective.reduction,
                               'metric': metric.reduction
                           }),
        RecordLearningRate(optimizer=optimizer, prefix="trainer"),
        ShowEpochProgress(header="Epoch {epoch_index}/{num_epochs}",
                          plugins=["epoch_index", "num_epochs"],
                          field_formatter=RegularReplacer({
                              r"trainer/lr": "lr",
                              r"trainer/metric/(.*)": r"\1",
                              r"trainer/objective": r"loss"
                          }),
                          value_format_rules=[
                              (WildcardMatcher("time/batch/*"),
                               TimeNumberFormatter()),
                              (None, TemplateNumberFormatter("{:.3g}"))
                          ],
                          max_field_length=6),
        EvaluateDatasets(inferer,
                         inferer_data_loaders,
                         plugins=["dataset", "num_epochs", "epoch_index"],
                         use_zeroth_epoch=not args.skip_initial_evaluation),
        RecordTimeMeters(time_meters, prefix="trainer"),
    ]
    if args.recorder:
      record_weights = 'weight' in args.recorder
      record_gradients = 'gradient' in args.recorder
      record_gradient_norm = 'gradient_norm' in args.recorder
      trainer_callbacks.append(
          RecordModelWeightsPerEpoch(prefix='trainer',
                                     record_weights=record_weights,
                                     record_gradients=record_gradients,
                                     record_gradient_norm=record_gradient_norm))

    if hasattr(config.data_loaders, "rayleigh"):
      from utils.callbacks.rayleigh import \
        RecordCrossEntropyGaussNewtonRayleighAndGradientAngle
      rayleigh_fields = ['rayleigh_global', 'rayleigh_local']
      for field in rayleigh_fields:
        num_batches = getattr(config.datasets, field).num_batches
        frequency = getattr(config.datasets, field).frequency
        setattr(
            config.datasets, field,
            build_dataset(getattr(config.datasets, field),
                          config.data_loaders.rayleigh,
                          config.datasets.transforms.training,
                          args.dataset_path,
                          num_workers=args.num_workers,
                          use_cuda=bool(args.gpus),
                          shuffle=True,
                          training_dataset=config.datasets.training))
        trainer_callbacks.append(
            RecordCrossEntropyGaussNewtonRayleighAndGradientAngle(
                getattr(config.datasets, field).data_loader,
                num_batches=num_batches,
                frequency=frequency,
                prefix=f"trainer/{remove_prefix(field, 'rayleigh_')}"))

    trainer_callbacks.extend([
        SynchronizeSummary(use_zeroth_epoch=not args.skip_initial_evaluation),
        ApplyPerEpochLRScheduler(scheduler)
    ])

    if "logger" not in args.disable_summary_writer:
      trainer_callbacks.append(
          WriteSummaryToLogger(
              use_zeroth_epoch=not args.skip_initial_evaluation,
              value_format_rules=[
                  (WildcardMatcher(['*/time/*',
                                    '*/time/batch/*']), TimeNumberFormatter()),
                  (WildcardMatcher('*/metric/top_*'),
                   TemplateNumberFormatter('{:.3g}')),
                  (WildcardMatcher('*'), TemplateNumberFormatter('{:.7g}'))
              ],
          ))
    else:
      logger.info("Disabled writing summary to logger")

    if "json" not in args.disable_summary_writer:
      trainer_callbacks.append(
          WriteSummaryToJSON(use_zeroth_epoch=not args.skip_initial_evaluation,
                             path=os.path.join(session_path, "tables")))
    else:
      logger.info("Disabled writing summary to JSON files")

    if "tensorboard" not in args.disable_summary_writer:
      trainer_callbacks.append(
          WriteSummaryToTensorBoard(
              use_zeroth_epoch=not args.skip_initial_evaluation,
              path=session_path))
    else:
      logger.info("Disabled writing summary to TensorBoard")

    if getattr(config.checkpoint, 'save_on_better_metric', None):
      comparator = getattr(config.checkpoint.save_on_better_metric,
                           "comparator", "max")
      callback = SaveCheckpointOnBetterMetric(
          metric_field=config.checkpoint.save_on_better_metric.field,
          comparator=comparator)
      trainer_callbacks.append(callback)

    trainer: SimpleTorchTrainer = checkpoint.attach(
        "trainer",
        SimpleTorchTrainer(model=model,
                           optimizer=optimizer,
                           objective=objective,
                           metric=metric,
                           callback=ComposeCallback(trainer_callbacks),
                           device=args.default_device))

    # Start training
    try:
      if args.resume_path:
        checkpoint.load(path=args.resume_path,
                        kwargs={"model": {
                            "map_location": args.default_device
                        }})
        trainer.run(config.datasets.training.data_loader)
      else:
        trainer.run(config.datasets.training.data_loader,
                    num_epochs=config.num_epochs)
    except KeyboardInterrupt:
      logger.warning("The training process is terminated by the user")
    except:
      logger.exception("An exception happened during training")


def parse_args():
  formatter_class = functools.partial(argparse.HelpFormatter,
                                      max_help_position=32)
  parser = argparse.ArgumentParser(formatter_class=formatter_class)

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("-C",
                     "--config_path",
                     type=str,
                     metavar="PATH",
                     default=None,
                     help="path to the training config")
  group.add_argument("-R",
                     "--resume_path",
                     type=str,
                     metavar="PATH",
                     default=None,
                     help="path to the session/model to be resumed")
  parser.add_argument("-D",
                      "--dataset_path",
                      type=str,
                      required=True,
                      metavar="PATH",
                      help="path to the base folder of datasets")
  parser.add_argument("-S",
                      "--session_path",
                      type=str,
                      default="./sessions",
                      metavar="PATH",
                      help="path to the base folder of sessions")
  parser.add_argument("--cache_path",
                      type=str,
                      default="./cache",
                      metavar="PATH",
                      help="path to the base folder for caching files")
  parser.add_argument("-G",
                      "--gpus",
                      action="store_true",
                      help="use GPUs for training acceleration")
  parser.add_argument("-N",
                      "--num_workers",
                      type=int,
                      default=os.cpu_count(),
                      help="the number of data loaders workers")
  parser.add_argument("--distribute",
                      action="store_true",
                      help="enable distributed mode")
  parser.add_argument("--seed",
                      type=int,
                      default=None,
                      help="random seed for reproducibility")
  parser.add_argument("--skip_initial_evaluation",
                      action="store_true",
                      help="skip the initial evaluation of the model")
  parser.add_argument('--recorder',
                      action="append",
                      metavar="FIELD",
                      default=[],
                      choices=['weight', 'gradient', 'gradient_norm'],
                      help="record values at the end of each epoch")
  parser.add_argument("--meter",
                      type=str,
                      action="append",
                      metavar="SCOPE",
                      default=[],
                      choices=['batch', 'forward', 'backward', 'update'],
                      help="add a time meter during training")
  parser.add_argument("--disable_summary_writer",
                      action="append",
                      metavar="WRITER",
                      default=[],
                      choices=['tensorboard', 'logger', 'json'],
                      help="disable a summary writer")
  args = parser.parse_args()

  args.gpus = build_gpus(args.gpus, args.distribute)
  if args.gpus:
    args.default_device = args.gpus[0]
    torch.cuda.set_device(args.gpus[0])
  else:
    args.default_device = torch.device('cpu')

  args.backend = None
  if args.distribute:
    args.backend = "nccl" if args.gpus else "gloo"
  return args


if __name__ == "__main__":
  main(parse_args())
