"""Common declartions."""

from typing import Optional, Dict, Any, Sequence, List

import dataclasses
import pathlib
import inspect
import shutil

import numpy as np
import jax.tree_util as tu

from dataclasses_json import dataclass_json

ClientState = Dict[str, Any]
Message = Optional[Dict[str, Any]]


@dataclass_json
@dataclasses.dataclass(frozen=True)
class VarianceSetting:
  group_correction: bool
  robust: bool
  cox_correction: bool


@dataclass_json
@dataclasses.dataclass(frozen=True)
class Config:
  """Variance configuration"""

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


def get_vars(state, *names):
  return tuple(lambda n: state[n] for n in names)


@dataclasses.dataclass
class ClientState:
  config: Config
  state: Dict[str, np.ndarray] = dataclasses.field(default=dict)

  def __getitem__(self, k):
    return self.state[k]

  def __setitem__(self, k, v):
    return self.state.__setitem__(k, v)

  def update(self, vals):
    return self.state.update(vals)

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
    in_msgs = [
        np.load(msg_file)
        for msg_file in client_dir.joinpath("in_msgs").glob("*.npz")
    ]
    if in_msgs:
      in_msgs = tu.tree_multimap(lambda args: np.array(*args), *in_msgs)
      client_state.update(in_msgs)

    # Do processing
    out_msg = fun(client_state, *args, **kwargs)

    # Remove old messages
    for msg in client_dir.joinpath("out_msgs").glob("*.npz"):
      shutil.rmtree(msg)

    # Send out msgs
    if out_msg:
      num_targets = [len(v) for v in out_msg.values() if isinstance(v, list)]
      if len(set(num_targets)) > 1:
        raise ValueError("Num targets does not match")
      num_targets = num_targets if num_targets else 1

      out_msgs = [{} for _ in range(num_targets)]
      for k, v in out_msg.items():
        if not isinstance(v, list):
          v = [v] * len(out_msgs)
        for i, (vi, out_msg) in enumerate(zip(v, out_msgs)):
          out_msg[k] = vi

      for i, out_msg in enumerate(out_msgs):
        out_msg_file = client_dir.joinpath("out_msgs", "msg_{}.npz".format(i))
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
