"""Common declartions."""

from typing import Optional, Dict, Any, List

import dataclasses
import pathlib
import inspect
import os
import re

import numpy as np
import jax.tree_util as tu

from dataclasses_json import dataclass_json

Message = Optional[Dict[str, Any]]


@dataclass_json
@dataclasses.dataclass(frozen=True)
class VarianceSetting:
  group_correction: bool
  robust: bool
  cox_correction: bool

  def __str__(self):
    return "cov:{}group_correction|{}robust|{}cox_correction".format(
        *[("" if f else "no_")
          for f in [self.group_correction, self.robust, self.cox_correction]])


@dataclass_json
@dataclasses.dataclass(frozen=True)
class Config:
  """Variance configuration"""

  client_id: str

  taylor_order: int = 1

  variance_settings: List[VarianceSetting] = dataclasses.field(
      default_factory=list)

  def require(self, key: str) -> bool:
    return any(map(lambda s: getattr(s, key), self.variance_settings))

  @property
  def require_group_correction(self):
    return self.require("group_correction")

  @property
  def require_cox_correction(self):
    return self.require("cox_correction")

  @property
  def require_robust(self):
    return self.require("group_correction")


@dataclasses.dataclass
class ClientState:
  config: Config
  state: Dict[str, np.ndarray] = dataclasses.field(default=lambda: {'covs': {}})

  def __getitem__(self, k):
    return self.state[k]

  def __setitem__(self, k: str, v):
    if k.endswith("*"):
      for i, vi in enumerate(v):
        self.state.__setitem__(k[:-1] + str(i), vi)
      return None
    return self.state.__setitem__(k, v)

  def __delitem__(self, k: str):
    if k.endswith("*"):
      for k1 in self.state:
        if k1.startswith(k[:-1]):
          del self.state[k1]
    else:
      del self.state[k]

  def __contains__(self, k: str):
    return self.state.__contains__(k)

  def update(self, vals):
    return self.state.update(vals)

  def pop(self, k):
    return self.state.pop(k)

  def get_var(self, k):
    if k.endswith("*"):
      ret = [(n, self.get_var(n))
             for n in self.state.keys()
             if n.startswith(k[:-1])]
      ret = sorted(ret, key=lambda x: int(x[0][len(k) - 1:]))
      return tuple(r[1] for r in ret)
    return self.state[k]

  def get_vars(self, *keys):
    return tuple(self.get_var(k) for k in keys)

  @property
  def is_master(self):
    return self.config.client_id == "master"

  @property
  def is_local(self):
    return not self.is_master

  @staticmethod
  def load(config_path, client_state_path):
    with open(config_path, "r") as f:
      t = f.read()
      config = Config.from_json(t)  # pylint: disable=no-member
    state = dict(np.load(client_state_path))
    return ClientState(config, state)

  def save(self, client_state_path):
    np.savez(client_state_path, **self.state)


def raise_to_command(fun):
  """Raise a `fun' to a command.

  This is a higher order function that takes in a function:
     `[ClientState, *args, **kwargs] -> Message`.
  Effectively, this does the heavy lifting of
    1. load the client state (master or local)
    2. read any received messages and incorprate the messages into client state
    3. do the processing of `fun`
    4. clear the previous outbox and store the out message(s)
    5. store back the client state
  If fun has more than the sole argument ClientState, the rest of the arguments
  are propogated to the wrapped fun.
  """

  def wrapped(client_dir: pathlib.Path, *args, **kwargs):
    # Restore client state
    client_state_file = client_dir.joinpath("client_state.npz")
    config_file = client_dir.joinpath("config.json")
    client_state = ClientState.load(config_file, client_state_file)

    # Recieve msgs
    in_msgs = [(re.match(r"msg_from_(.+)_to_(.+).npz",
                         str(os.path.basename(msg_file))).group(1),
                dict(np.load(msg_file))) for msg_file in client_dir.joinpath(
                    "in_msgs").glob("msg_from_*_to_*.npz")]
    in_msgs = sorted(in_msgs, key=lambda t: t[0])
    if client_state.is_master:
      in_senders = [t[0] for t in in_msgs]
    else:
      in_senders = ["master"]
    in_msgs = [t[1] for t in in_msgs]
    if in_msgs:
      if len(in_msgs) > 1:

        def possibily_compact_array(*arrays):
          # Creates an np.array by stacking together `arrays`
          # If `arrays` have different shapes, create a ragged object array
          # This is to emulate the old numpy behavior prior to NEP34:
          #  https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
          if len(set(map(lambda a: a.shape, arrays))) == 1:
            return np.array(arrays)
          return np.array(arrays, dtype=object)

        in_msgs = tu.tree_multimap(possibily_compact_array, *in_msgs)
      else:
        in_msgs = in_msgs[0]
      client_state.update(in_msgs)

    # Do processing
    out_msg = fun(client_state, *args, **kwargs)

    # Remove old messages
    for msg in client_dir.joinpath("out_msgs").glob("*.npz"):
      os.remove(msg)

    # Send out msgs
    if out_msg:
      num_targets = [len(v) for v in out_msg.values() if isinstance(v, list)]
      if len(set(num_targets)) > 1:
        raise ValueError("Num targets does not match")

      num_targets = len(in_senders) if client_state.is_master else 1

      out_msgs = [{} for _ in range(num_targets)]
      for k, v in out_msg.items():
        if not isinstance(v, list):
          v = [v] * num_targets
        for i, (vi, out_msg) in enumerate(zip(v, out_msgs)):
          out_msg[k] = vi
      for i, (sender, out_msg) in enumerate(zip(in_senders, out_msgs)):
        out_msg_file = client_dir.joinpath(
            "out_msgs",
            "msg_from_{}_to_{}.npz".format(client_state.config.client_id,
                                           sender))
        np.savez(out_msg_file, **out_msg)

    # Save new client state
    client_state.save(client_state_file)

  # Some magic to propogate the wrapped function's signature
  wsig = inspect.signature(wrapped)
  wrapped.__name__ = fun.__name__
  sig = inspect.signature(fun)
  wrapped.__signature__ = sig.replace(
      parameters=tuple(wsig.parameters.values())[:1] +
      tuple(sig.parameters.values())[1:])

  return wrapped
