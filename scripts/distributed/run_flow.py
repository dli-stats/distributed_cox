"""Run Distributed flow."""

import subprocess
import sys
import os
import shutil
import pathlib
import tempfile
import logging
import json
import itertools

CMD = [sys.executable, "-W", "ignore", "-m", "distributed_cox.distributed.cmd"]

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
  return subprocess.check_call(cmd)


def run_flow(eq: str, N: int, K: int, X_DIM: int, seed: int = 0):

  tmp_work_dir = tempfile.mkdtemp(prefix="cox_run_flow--",
                                  dir=pathlib.Path.cwd())
  work_dir = pathlib.Path(tmp_work_dir)
  logging.info("Starting Distributed Cox Flow at %s", work_dir)

  data_dir = work_dir.joinpath("data")
  os.makedirs(data_dir)
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

  call_cmd(
      "dummy_data_gen",
      N,
      K,
      X_DIM,
      "--seed={}".format(seed),
      "--save_dir={}".format(data_dir),
  )

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

  def call_cmd_and_send(client, cmd, *args):
    if client == "master":
      call_cmd(cmd, master_dir, *args)
    else:
      for k in range(K):
        call_cmd(cmd, local_dirs[k], *args)
    copy_msgs(client)

  if eq == "eq2":
    call_cmd_and_send("local", "eq2_local_send_T")
    call_cmd_and_send("master", "eq2_master_send_T")
    call_cmd_and_send("local", "eq2_local")
    call_cmd_and_send("master", "eq2_master")
    call_cmd_and_send("local", "eq2_local_variance")
    call_cmd_and_send("master", "eq2_master_all_variances")
  if eq == "eq4":
    call_cmd_and_send("local", "eq4_local")
    call_cmd_and_send("master", "eq4_master")
    call_cmd_and_send("local", "eq4_local_variance")
    call_cmd_and_send("master", "eq4_master_all_variances")


run_flow("eq4", 1000, 3, 3, 100)
