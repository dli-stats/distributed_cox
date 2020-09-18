"""Commandline for distributed."""

from typing import Optional

import pathlib
import os
import shutil

import numpy as np

import typer

import jax.random as jrandom

import distributed_cox.data as vdata

from distributed_cox.distributed.common import (Config, VarianceSetting,
                                                ClientState, raise_to_command)
import distributed_cox.distributed.eq2 as eq2

app = typer.Typer(add_completion=False)


@app.command()
def init_client(client_dir: pathlib.Path, force: bool = False):
  """Initializes a client with necessary directory creations."""
  if force:
    shutil.rmtree(client_dir)
  os.makedirs(client_dir, exist_ok=False)
  os.makedirs(client_dir.joinpath("in_msgs"))
  os.makedirs(client_dir.joinpath("out_msgs"))
  default_config = Config(
      taylor_order=1, variance_settings=[VarianceSetting(False, False, False)])
  with open(client_dir.joinpath("config.json"), "w+") as f:
    f.write(default_config.to_json())  # pylint: disable=no-member
  np.savez(client_dir.joinpath("client_state.npz"))


def local_load_data(local_state: ClientState, data_path: pathlib.Path):
  data = np.load(data_path)
  local_state.update(data)
  return {}


@app.command()
def dummy_data_gen(n: int,
                   k: int,
                   x_dim: int,
                   T_star_factors: Optional[str] = None,
                   group_labels_generator_kind: str = "same",
                   group_X: str = "same",
                   exp_scale: float = 3.5,
                   save_dir: pathlib.Path = pathlib.Path.cwd,
                   seed: int = 0):
  """Generates Some dummy cox data."""
  N, K, X_DIM = n, k, x_dim
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


to_cmd = lambda f: app.command()(raise_to_command(f))

to_cmd(local_load_data)
to_cmd(eq2.eq2_local_send_T)
to_cmd(eq2.eq2_master_send_T)
to_cmd(eq2.eq2_local)
to_cmd(eq2.eq2_master)
to_cmd(eq2.eq2_local_variance)
to_cmd(eq2.eq2_master_variance)

if __name__ == "__main__":
  app()
