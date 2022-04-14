__all__ = ['TorchGPUTimeMeter', 'TorchCPUTimeMeter']

import time
from typing import Collection, Optional

import torch

from flexlearn.meters import TimeMeter


class TorchGPUTimeMeter(TimeMeter):

  def __init__(self,
               cuda_device: Optional[torch.device] = None,
               cuda_stream: Optional[torch.cuda.Stream] = None):
    self._prior_event = torch.cuda.Event(enable_timing=True)
    self._after_event = torch.cuda.Event(enable_timing=True)
    self._cuda_stream = cuda_stream or torch.cuda.current_stream(cuda_device)
    self._elapsed_time = 0

  def prior(self):
    self._cuda_stream.synchronize()
    self._prior_event.record(stream=self._cuda_stream)

  def after(self):
    self._after_event.record(stream=self._cuda_stream)
    self._cuda_stream.synchronize()
    self._elapsed_time += self._prior_event.elapsed_time(self._after_event)

  def reset(self):
    self._elapsed_time = 0

  def elapsed_time(self):
    return self._elapsed_time


class TorchCPUTimeMeter(TimeMeter):

  def __init__(self, devices: Optional[Collection[torch.device]] = None):
    self._devices = list(devices) if devices is not None else list()
    self._elapsed_ns = 0

  def prior(self):
    for device in self._devices:
      if device.type == 'cuda':
        torch.cuda.synchronize(device)
    self._elapsed_ns -= time.time_ns()

  def reset(self):
    self._elapsed_ns = 0

  def after(self):
    for device in self._devices:
      if device.type == 'cuda':
        torch.cuda.synchronize(device)
    self._elapsed_ns += time.time_ns()

  def elapsed_time(self):
    return self._elapsed_ns / 1e6
