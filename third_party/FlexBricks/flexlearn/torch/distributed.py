__all__ = ['TorchProcessGroup']

import contextlib
import os
from typing import Optional, List, ByteString, Generator

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from flexlearn.distributed import ProcessGroup
from flexutils.misc import dict_lookup


class TorchProcessGroup(ProcessGroup):
  __root__: Optional['TorchProcessGroup'] = None

  def __init__(self,
               handle: Optional[dist.ProcessGroup],
               *,
               device: Optional[torch.device] = None,
               global_rank: Optional[int] = None,
               global_size: Optional[int] = None,
               local_rank: Optional[int] = None,
               local_size: Optional[int] = None,
               node_rank: Optional[int] = None,
               node_size: Optional[int] = None,
               validate: bool = True):
    self._handle = handle
    if self.is_active():
      if global_rank is None:
        global_rank = dist.get_rank(handle)
      if global_size is None:
        global_size = dist.get_world_size(handle)
      if device is None:
        handle = self._handle or dist.distributed_c10d._get_default_group()
        if dist.is_nccl_available() and isinstance(handle,
                                                   dist.ProcessGroupNCCL):
          device = torch.device('cuda', torch.cuda.current_device())
        else:
          device = torch.device('cpu')
      self._device = device
    else:
      self._device = None
    super().__init__(global_rank=global_rank,
                     global_size=global_size,
                     local_rank=local_rank,
                     local_size=local_size,
                     node_rank=node_rank,
                     node_size=node_size,
                     validate=validate)

  def close(self):
    if self._handle is not None:
      dist.destroy_process_group(self._handle)
    self._handle = dist.GroupMember.NON_GROUP_MEMBER

  @property
  def handle(self):
    return self._handle

  @property
  def device(self):
    return self._device

  @classmethod
  def subgroup(cls, ranks=None, *args, **kwargs):
    device = kwargs.pop("device", cls.__root__.device)
    handle = dist.new_group(ranks=ranks, *args, **kwargs)
    return TorchProcessGroup(handle, device=device)

  def is_active(self):
    return self._handle != dist.GroupMember.NON_GROUP_MEMBER

  def all_gather_same_bytes(self, data: ByteString, **kwargs) \
          -> Optional[List[bytes]]:
    if not self.is_active():
      return

    size = len(data)
    # noinspection PyUnresolvedReferences
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(data))
    data = data.to(self._device)
    rank_data = torch.empty(size * self._global_size,
                            dtype=torch.uint8,
                            device=self._device)
    rank_data = list(rank_data.split(size))
    dist.all_gather(rank_data, data, group=self._handle)
    for i in range(len(rank_data)):
      rank_data[i] = rank_data[i].cpu().numpy().tobytes()
    return rank_data

  def all_gather_bytes(self, data: ByteString, **kwargs):
    if not self.is_active():
      return

    size = torch.LongTensor([len(data)])
    size = size.to(self._device)
    rank_size = torch.empty(self._global_size,
                            dtype=torch.long,
                            device=self._device)
    dist.all_gather(list(rank_size.split(1)), size, group=self._handle)
    max_size = int(rank_size.max().item())

    # noinspection PyUnresolvedReferences
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(data))
    data = data.to(self._device)
    data.resize_(max_size)

    rank_data = torch.empty(max_size * self._global_size,
                            dtype=torch.uint8,
                            device=self._device)
    rank_data = list(rank_data.split(max_size))
    dist.all_gather(rank_data, data, group=self._handle)
    for i in range(len(rank_data)):
      rank_data[i] = rank_data[i].cpu()
      rank_data[i] = rank_data[i][:rank_size[i]].numpy().tobytes()
    return rank_data

  def all_gather_tensor(self,
                        tensor_list: list,
                        tensor: torch.Tensor,
                        use_async_op: bool = False):
    return dist.all_gather(tensor_list,
                           tensor,
                           group=self._handle,
                           async_op=use_async_op)

  def all_reduce_tensor(self,
                        tensor: torch.Tensor,
                        op: dist.ReduceOp = dist.ReduceOp.SUM,
                        use_async_op: bool = False):
    return dist.all_reduce(tensor,
                           op,
                           group=self._handle,
                           async_op=use_async_op)

  def all_reduce_tensor_multiple_gpu(self,
                                     tensors: Optional[torch.Tensor],
                                     op: dist.ReduceOp = dist.ReduceOp.SUM,
                                     use_async_op: bool = False):
    return dist.all_reduce_multigpu(tensors,
                                    op,
                                    group=self._handle,
                                    async_op=use_async_op)

  def all_reduce_tensor_coalesced(self,
                                  tensors: Optional[torch.Tensor],
                                  op: dist.ReduceOp = dist.ReduceOp.SUM,
                                  use_async_op: bool = False):
    return dist.all_reduce_coalesced(tensors,
                                     op,
                                     group=self._handle,
                                     async_op=use_async_op)

  def broadcast_same_bytes(self, data: ByteString, source, **kwargs):
    if not self.is_active():
      return

    # noinspection PyUnresolvedReferences
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(data))
    data = data.to(self._device)
    dist.broadcast(data, source, group=self._handle)
    if data.device != torch.device("cpu"):
      data = data.cpu()
    return data.numpy().tobytes()

  def broadcast_bytes(self, data: ByteString, source: int, **kwargs):
    if not self.is_active():
      return

    size = torch.LongTensor([len(data)])
    size = size.to(self._device)
    dist.broadcast(size, source, group=self._handle)

    # noinspection PyUnresolvedReferences
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(data))
    data = data.to(self._device)
    data.resize_(size)
    dist.broadcast(data, source, group=self._handle)
    if data.device != torch.device("cpu"):
      data = data.cpu()

    return data.numpy().tobytes()

  def broadcast_tensor(self,
                       tensor: torch.Tensor,
                       source: int,
                       use_async_op: bool = False):
    return dist.broadcast(tensor,
                          source,
                          group=self._handle,
                          async_op=use_async_op)

  def barrier(self, device_ids=None, use_async_op: bool = False, **kwargs):
    return dist.barrier(self._handle,
                        device_ids=device_ids,
                        async_op=use_async_op)

  def parallelize(self, module: torch.nn.Module, **kwargs):
    if not self.is_active():
      return
    return DistributedDataParallel(module, process_group=self._handle, **kwargs)

  @classmethod
  def is_activated(cls):
    return cls.__root__ is not None

  @classmethod
  def root(cls) -> 'TorchProcessGroup':
    return cls.__root__

  @classmethod
  def env_local_rank(cls):
    return dict_lookup(os.environ, "LOCAL_RANK")

  @classmethod
  def env_local_size(cls):
    return dict_lookup(os.environ, "LOCAL_SIZE", "LOCAL_WORLD_SIZE")

  @classmethod
  def env_node_rank(cls):
    return dict_lookup(os.environ, "NODE_RANK", "GROUP_RANK")

  @classmethod
  def env_node_size(cls):
    return dict_lookup(os.environ, "NODE_SIZE", "GROUP_WORLD_SIZE")

  @classmethod
  def env_global_rank(cls):
    return dict_lookup(os.environ, "GLOBAL_RANK", "RANK")

  @classmethod
  def env_global_size(cls):
    return dict_lookup(os.environ, "GLOBAL_SIZE", "WORLD_SIZE")

  @classmethod
  def env_master_addr(cls):
    return dict_lookup(os.environ, "MASTER_ADDR")

  @classmethod
  def env_master_port(cls):
    return dict_lookup(os.environ, "MASTER_PORT")

  @classmethod
  @contextlib.contextmanager
  def activate(cls,
               backend: Optional[str] = None,
               global_rank: Optional[int] = None,
               global_size: Optional[int] = None,
               local_rank: Optional[int] = None,
               local_size: Optional[int] = None,
               node_rank: Optional[int] = None,
               node_size: Optional[int] = None,
               validate=True,
               device: Optional[torch.device] = None,
               **kwargs) -> Generator['TorchProcessGroup', None, None]:
    if cls.__root__ is not None:
      raise RuntimeError(f"{cls.__name__} has already been activated")
    if backend is None:
      yield
      return

    try:
      if global_rank is None:
        global_rank = cls.env_global_rank()
      if global_rank is not None:
        global_rank = int(global_rank)

      if global_size is None:
        global_size = cls.env_global_size()
      if global_size is not None:
        global_size = int(global_size)

      if local_rank is None:
        local_rank = cls.env_local_rank()
      if local_rank is not None:
        local_rank = int(local_rank)

      if local_size is None:
        local_size = cls.env_local_size()
      if local_size is not None:
        local_size = int(local_size)

      if node_rank is None:
        node_rank = cls.env_node_rank()
      if node_rank is not None:
        node_rank = int(node_rank)
      else:
        raise RuntimeError(f"{cls.__name__} requires node_rank to be specified")

      if node_size is None:
        node_size = cls.env_node_size()
      if node_size is not None:
        node_size = int(node_size)

      dist.init_process_group(backend=backend, **kwargs)
      cls.__root__ = TorchProcessGroup(None,
                                       device=device,
                                       global_rank=global_rank,
                                       global_size=global_size,
                                       local_rank=local_rank,
                                       local_size=local_size,
                                       node_rank=node_rank,
                                       node_size=node_size,
                                       validate=validate)
      yield cls.__root__
    finally:
      dist.destroy_process_group()
      cls.__root__ = None
