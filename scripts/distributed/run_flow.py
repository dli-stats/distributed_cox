"""Run distributed flow driver."""

from typing import Callable

import subprocess
import sys
import os
import shutil
import pathlib
import tempfile
import logging
import json
import itertools
import code

import numpy as onp

from distributed_cox.simple_cmd import SimpleCMD

cmd = SimpleCMD()

CMD = [sys.executable, "-m", "distributed_cox.distributed.cmd"]

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

variance_settings = []
for group_correction, robust, cox_correction in itertools.product(
    *[[True, False]] * 3):
  if cox_correction and not robust:
    continue
  variance_settings.append(
      dict(group_correction=group_correction,
           robust=robust,
           cox_correction=cox_correction))


def call_cmd(*args):
  cmd = CMD + [str(arg) for arg in args]
  logging.info("Running command: %s", " ".join(cmd))
  return subprocess.check_call(cmd, env={"JAX_PLATFORM_NAME": "cpu"})


def _run_flow(fill_data_fn: Callable,
              method: str,
              check_result: bool = False,
              seed: int = 0) -> pathlib.Path:
  del seed

  tmp_work_dir = tempfile.mkdtemp(prefix="cox_run_flow--",
                                  dir=pathlib.Path.cwd())
  work_dir = pathlib.Path(tmp_work_dir)
  logging.info("Starting Distributed Cox Flow at %s", work_dir)

  data_dir = work_dir.joinpath("data")
  os.makedirs(data_dir)

  _, K, _ = fill_data_fn(data_dir)

  master_dir = work_dir.joinpath("master")
  local_dirs = [work_dir.joinpath(str(k)) for k in range(K)]

  def copy_msgs(from_client="master"):
    if from_client == "master":
      for k in range(K):
        msg = "msg_from_master_to_{}.npz".format(k)
        shutil.move(master_dir.joinpath("out_msgs", msg),
                    local_dirs[k].joinpath("in_msgs", msg))
    else:
      for k in range(K):
        msg = "msg_from_{}_to_master.npz".format(k)
        shutil.move(local_dirs[k].joinpath("out_msgs", msg),
                    master_dir.joinpath("in_msgs", msg))

  # Init clients
  def init_client(client_dir: pathlib.Path, client: str):
    call_cmd("init_client", client_dir, client)
    with open(client_dir.joinpath("config.json"), "r") as f:
      config = json.load(f)
    config["variance_settings"] = variance_settings
    with open(client_dir.joinpath("config.json"), "w") as f:
      json.dump(config, f)

  init_client(master_dir, "master")
  for k in range(K):
    init_client(local_dirs[k], str(k))

  # Load data
  for k in range(K):
    call_cmd(
        "local_load_data",
        local_dirs[k],
        data_dir.joinpath("local_{}.npz".format(k)),
    )

  def call_cmd_and_send(client, cmd, *args, end=False):
    if client == "master":
      call_cmd(cmd, master_dir, *args)
    else:
      for k in range(K):
        call_cmd(cmd, local_dirs[k], *args)
    if not end:
      copy_msgs(client)

  if method == "unstratified_distributed":
    call_cmd_and_send("local", "unstratified_distributed_local_send_T")
    call_cmd_and_send("master", "unstratified_distributed_master_send_T")
    call_cmd_and_send("local", "unstratified_distributed_local")
    call_cmd_and_send("master", "unstratified_distributed_master")
    call_cmd_and_send("local", "unstratified_distributed_local_variance")
    call_cmd_and_send("master",
                      "unstratified_distributed_master_all_variances",
                      end=True)
  if method == "stratified_distributed":
    call_cmd_and_send("local", "stratified_distributed_local")
    call_cmd_and_send("master", "stratified_distributed_master")
    call_cmd_and_send("local", "stratified_distributed_local_variance")
    call_cmd_and_send("master",
                      "stratified_distributed_master_all_variances",
                      end=True)

  if check_result:
    client = onp.load(work_dir.joinpath("master", "client_state.npz"))  # pylint: disable=unused-variable
    code.interact(local=dict(client=client))

  return work_dir


@cmd.command
def run_dummy_flow(method: str,
                   N: int,
                   K: int,
                   X_DIM: int,
                   check_result: bool = False,
                   seed: int = 0) -> pathlib.Path:
  """Generate dummy data then run a distributed flow."""

  def fill_data(data_dir: pathlib.Path):
    call_cmd(
        "dummy_data_gen",
        N,
        K,
        X_DIM,
        "--seed={}".format(seed),
        "--save_dir={}".format(data_dir),
    )
    return N, K, X_DIM

  return _run_flow(fill_data, method, check_result, seed)


@cmd.command
def run_flow(method: str,
             data_dir: pathlib.Path,
             check_result: bool = False,
             seed: int = 0) -> pathlib.Path:

  def fill_data(flow_data_dir: pathlib.Path):
    all_data = onp.load(data_dir.joinpath("all.npz"))
    N, X_DIM = all_data['X'].shape
    K = onp.max(all_data["group_labels"]) + 1
    names = ["all.npz"] + [f"local_{i}.npz" for i in range(K)]
    for name in names:
      shutil.copy(data_dir.joinpath(name), flow_data_dir.joinpath(name))
    return N, K, X_DIM

  return _run_flow(fill_data, method, check_result, seed)


if __name__ == "__main__":
  cmd.run()
