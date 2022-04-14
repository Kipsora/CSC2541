__all__ = ['ProcessGroup']

import abc
from typing import Optional, Any, Dict, ByteString, List

from flexutils.io import serialize


class ProcessGroup(object, metaclass=abc.ABCMeta):
  """
  :class:`ProcessGroup` unifies the interface of different distributed
  communication protocol or implementations. Currently, we only have the torch
  implementation (see :class:`~flexlearn.torch.distributed.TorchProcessGroup`)

  Args:
    global_rank: The global rank of the current process.
    global_size: The global size of the current process.
    local_rank: The local rank of the current process. This argument is optional
      and can be automatically deduced using :meth:`all_gather` if not given.
    local_size: The local size of the current process. This argument is optional
      and can be automatically deduced using :meth:`all_gather` if not given.
    node_rank: The node rank of the current process. This argument is optional
      and can be automatically deduced using :meth:`all_gather` if not given.
    node_size: The node rank of the current process. This argument is optional
      and can be automatically deduced using :meth:`all_gather` if not given.
    validate: Whether to validate the given arguments.
  """

  __default__: Optional['ProcessGroup'] = None

  def __init__(self,
               *,
               global_rank: int,
               global_size: int,
               local_rank: Optional[int] = None,
               local_size: Optional[int] = None,
               node_rank: Optional[int] = None,
               node_size: Optional[int] = None,
               validate: bool = True):
    self._global_rank = global_rank
    self._global_size = global_size

    self._local_size = local_size
    self._local_rank = local_rank

    self._node_rank = node_rank
    self._node_size = node_size

    self._locals = None

    self._deduce_and_validate(validate)

  @classmethod
  def is_local_master(cls, process_group: Optional['ProcessGroup']) -> bool:
    """
    Whether the current process is the master process within the current node.

    Args:
      process_group: An instance of :class:`ProcessGroup`. If it is ``None``,
        then the method will always return ``True``.

    Returns:
      Whether the current process is the master process locally.
    """
    return process_group is None or process_group.local_rank() == 0

  @classmethod
  def is_global_master(cls, process_group: Optional['ProcessGroup']) -> bool:
    """
    Whether the current process is the master process within all nodes.

    Args:
      process_group: An instance of `ProcessGroup`
        if it is ``None``, then the method will always return ``True``.

    Returns:
      Whether the current process is the master process globally.
    """
    return process_group is None or process_group.global_rank() == 0

  def locals(self) -> List[int]:
    """
    Retrieve all global ranks that locate at the same node.

    Returns:
       A list of ranks.
    """
    if self._locals is None:
      self._locals = self.all_gather(self._node_rank)
      self._locals = [
          i for i, node_rank in enumerate(self._locals)
          if self._node_rank == node_rank
      ]
    return self._locals

  def _deduce_and_validate(self, validate: bool = True):
    if not self.is_active():
      return
    if self._node_rank is None or self._node_size is None or \
            self._local_rank is None or self._local_size is None or validate:
      if validate:
        self._validate_ranks(self.all_gather(self._global_rank), "global_rank")

      if self._node_rank is None:
        all_node_ranks = self.all_gather(self.root()._node_rank)
        new_node_ranks = sorted(set(all_node_ranks))
        new_node_ranks = {r: i for i, r in enumerate(new_node_ranks)}
        all_node_ranks = list(map(new_node_ranks.get, all_node_ranks))
        expected_node_size = len(new_node_ranks)
        self._node_rank = all_node_ranks[self._global_rank]
      else:
        all_node_ranks = self.all_gather(self._node_rank)
        self._validate_ranks(all_node_ranks, "node_rank")
        expected_node_size = len(set(all_node_ranks))
      self._locals = [
          i for i, node_rank in enumerate(all_node_ranks)
          if self._node_rank == node_rank
      ]

      if self._node_size is None:
        self._node_size = expected_node_size
      elif self._node_size != expected_node_size:
        raise ValueError(
            f"The node_size mismatches with the expected value "
            f"(given={self._node_size}, expected={expected_node_size})")

      if self._local_rank is None:
        self._local_rank = all_node_ranks[:self._global_rank]
        self._local_rank = self._local_rank.count(self._node_rank)
      else:
        if validate:
          self._validate_ranks(self.all_gather(self._local_rank), "local_rank")

      expected_local_size = all_node_ranks.count(self._node_rank)
      if self._local_size is None:
        self._local_size = expected_local_size
      elif self._local_size != expected_local_size:
        raise ValueError(
            f"The local_size mismatches with the expected value "
            f"(given={self._local_size}, expected={expected_local_size})")

  @classmethod
  @abc.abstractmethod
  def root(cls) -> Optional['ProcessGroup']:
    """
    The world group.

    Returns:
      A group instance or None if the downstream implementation has not
      been activated.
    """
    raise NotImplementedError

  @classmethod
  def _validate_ranks(cls, ranks: List[int], name: str):
    max_rank = max(ranks)
    if max_rank != len(set(ranks)) - 1:
      raise ValueError(f"Discontinuous {name} is found ("
                       f"global_rank={ranks.index(max_rank)}, "
                       f"{name}={max_rank})")

    min_rank = min(ranks)
    if min_rank != 0:
      raise ValueError(f"Discontinuous node rank is found ("
                       f"global_rank={ranks.index(min_rank)}, "
                       f"{name}={min_rank}")

  @abc.abstractmethod
  def close(self):
    """
    Close the group and release any possible resources. Note that the root
    group may not be freed as some it may be managed by a package vendor.
    """
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def global_size(self) -> int:
    """
    The number of peers that use the current group.

    Returns:
      The global size.
    """
    return self._global_size

  def global_rank(self) -> int:
    """
    The rank within all peers that use the current group.

    Returns:
      The global rank.
    """
    return self._global_rank

  def local_size(self) -> int:
    """
    The number of peers that use the current group and locate at
    the same node.

    Returns:
      The local size.
    """
    return self._local_size

  def local_rank(self) -> int:
    """
    The rank within all peers that use the current group and locate at the
    same node.

    Returns:
      The local rank.
    """
    return self._local_rank

  def node_size(self) -> int:
    """
    The number of nodes that use the current group.

    Returns:
      The node size.
    """
    return self._node_size

  def node_rank(self) -> int:
    """
    The rank within all nodes that use the current group.

    Returns:
      The node rank.
    """
    return self._node_rank

  @abc.abstractmethod
  def all_gather_bytes(self, data: ByteString, **kwargs) -> List[bytes]:
    """
    Perform the ``all_gather`` operation on data, assuming data is a
    :py:class:`~typing.ByteString`.

    Args:
      data: The data to be gathered.
      **kwargs: Implementation specific arguments.
    Returns:
      A :class:`list` that contains the result or ``None`` if inactive.
    """
    pass

  def all_gather_same_bytes(self, data: ByteString, **kwargs) -> List[bytes]:
    """
    Perform the all_gather operation on data, assuming data is a ByteString and
    has the same length across all peers. Depending on the implementation,
    this method can have better performance.

    Args:
      data: The data to be gathered.
      **kwargs: Implementation specific arguments.
    Returns:
      A :class:`list` that contains the result or ``None`` if inactive.
    """
    return self.all_gather_bytes(data, **kwargs)

  @abc.abstractmethod
  def broadcast_bytes(self, data: ByteString, source: int, **kwargs) -> bytes:
    """
    Perform broadcast operation on data, assuming data is a
    :py:class:`~typing.ByteString`.

    Args:
      data: The data to be broadcast.
      source: The (global) rank of the source peer.
      **kwargs: Implementation specific arguments.
    Returns:
      A byte string that contains the result or None if inactive.
    """
    pass

  @abc.abstractmethod
  def is_active(self) -> bool:
    """
    Whether the current group is active. Downstream implementations can have
    subgroup method which creates a subgroup that only communicates between a
    selected subset of peers. Peers that are not in this subset are marked as
    `inactive` and all operators on inactive peers should be ignored.

    Returns:
      Whether the current peer is active.
    """
    pass

  def all_gather(self,
                 data,
                 serializer: str = "pickle",
                 load_kwargs: Optional[Dict[str, Any]] = None,
                 dump_kwargs: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Optional[list]:
    """
    Perform the ``all_gather`` operation on data, which is a python object.
    Default implementation is to first serialize the data and then gathered with
    :meth:`all_gather_bytes` method.

    Args:
      data: The data to be gathered
      serializer: The serializer to serialize the python object.
      load_kwargs: Arguments of `load` which recovers the object
        from bytes.
      dump_kwargs: Arguments of `dump` which converts the object to bytes.
      **kwargs: Implementation specific arguments.
    Returns:
      A :class:`list` that contains the result or ``None`` if inactive.
    """
    if not self.is_active():
      return
    data = serialize.dumpb(data, serializer, **(dump_kwargs or dict()))
    data = [
        serialize.loadb(e, serializer, **(load_kwargs or dict()))
        for e in self.all_gather_bytes(data, **kwargs)
    ]
    return data

  def broadcast(self,
                data,
                source: int,
                serializer: str = "pickle",
                load_kwargs: Optional[Dict[str, Any]] = None,
                dump_kwargs: Optional[Dict[str, Any]] = None,
                **kwargs) -> Optional[list]:
    """
    Perform the broadcast operation on data, which is a python object. Default
    implementation is to first serialize the data and then gathered with
    broadcast_bytes method.

    Args:
      data: The data to be gathered
      source: The (global) rank of the source peer.
      serializer: The serializer to serialize the python object.
      load_kwargs: Arguments of `load` which recovers the object from bytes.
      dump_kwargs: Arguments of `dump` which converts the object to bytes.
      **kwargs: Implementation specific arguments.
    Returns:
      A :class:`list` that contains the result or ``None`` if inactive.
    """
    if not self.is_active():
      return
    data = serialize.dumpb(data, serializer, **(dump_kwargs or dict()))
    return serialize.loadb(self.broadcast_bytes(data, source, **kwargs),
                           serializer, **(load_kwargs or dict()))

  @abc.abstractmethod
  def barrier(self, **kwargs):
    """
    Perform the barrier operation. It will block the current thread until all
    peers have called this method. It will have no effect if inactive.

    Args:
      **kwargs: Implementation specific arguments.
    """
    pass
