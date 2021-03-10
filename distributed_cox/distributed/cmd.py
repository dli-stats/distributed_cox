"""Commandline for distributed."""

from typing import Optional

import pathlib
import os
import shutil

import absl.logging
# Disable JAX's CPU vs. GPU warning...
absl.logging.set_verbosity(absl.logging.FATAL)

# pylint: disable=wrong-import-position
import numpy as np

import jax.random as jrandom

import distributed_cox.simple_cmd as simple_cmd

import distributed_cox.data as vdata

from distributed_cox.distributed.common import (Config, VarianceSetting,
                                                ClientState, raise_to_command)
import distributed_cox.distributed.eq2 as eq2
import distributed_cox.distributed.eq4 as eq4

cmd = simple_cmd.SimpleCMD()


@cmd.command
def init_client(client_dir: pathlib.Path, client_id: str, force: bool = False):
  """Initializes a client with necessary directory creations."""
  if force:
    shutil.rmtree(client_dir, ignore_errors=True)
  os.makedirs(client_dir, exist_ok=False)
  os.makedirs(client_dir.joinpath("in_msgs"))
  os.makedirs(client_dir.joinpath("out_msgs"))
  default_config = Config(
      client_id=client_id,
      taylor_order=1,
      variance_settings=[VarianceSetting(False, False, False)])
  with open(client_dir.joinpath("config.json"), "w+") as f:
    f.write(default_config.to_json())  # pylint: disable=no-member
  np.savez(client_dir.joinpath("client_state.npz"))


def local_load_data(local_state: ClientState, data_path: pathlib.Path):
  data = np.load(data_path)
  local_state.update(data)
  return {}


@cmd.command
def dummy_data_gen(N: int,
                   K: int,
                   X_DIM: int,
                   T_star_factors: Optional[str] = None,
                   group_labels_generator_kind: str = "same",
                   group_X: str = "same",
                   exp_scale: float = 3.5,
                   save_dir: pathlib.Path = pathlib.Path.cwd(),
                   seed: int = 0):
  """Generates some dummy Cox data."""
  key = jrandom.PRNGKey(seed)
  _, gen_fn = vdata.full_data_generator(N, K, X_DIM, T_star_factors,
                                        group_labels_generator_kind, group_X,
                                        exp_scale)
  _, T, X, delta, _, group_labels = gen_fn(key)
  np.savez(save_dir.joinpath('all'),
           X=X,
           T=T,
           delta=delta,
           group_labels=group_labels)

  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    np.savez(save_dir.joinpath("local_" + str(k)),
             X_group=X_group,
             T_group=T_group,
             delta_group=delta_group)


to_cmd = lambda f: cmd.command(raise_to_command(f))

to_cmd(local_load_data)

# Eq2
to_cmd(eq2.eq2_local_send_T)
to_cmd(eq2.eq2_master_send_T)
to_cmd(eq2.eq2_local)
to_cmd(eq2.eq2_master)
to_cmd(eq2.eq2_local_variance)
to_cmd(eq2.eq2_master_all_variances)
# Eq 4
to_cmd(eq4.eq4_local)
to_cmd(eq4.eq4_master)
to_cmd(eq4.eq4_local_variance)
to_cmd(eq4.eq4_master_all_variances)

if __name__ == "__main__":
  cmd.run()
