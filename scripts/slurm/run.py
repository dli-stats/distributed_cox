"""Run experiments."""

from typing import Dict, List

import collections
import os
import subprocess
import textwrap
import json
import hashlib
import math
import datetime
import argparse
from importlib import util

import numpy as np

import simple_slurm

root_dir = subprocess.check_output(["git", "rev-parse",
                                    "--show-toplevel"]).strip().decode("utf-8")

parser = argparse.ArgumentParser()
parser.add_argument("--storage_dir",
                    type=str,
                    default="/n/scratch3/users/d/dl263/varderiv_experiments")
parser.add_argument("--logs_dir", type=str, default="logs")
parser.add_argument(
    "--config_dir",
    type=str,
    default="/n/scratch3/users/d/dl263/varderiv_experiments_configs")
parser.add_argument("--n_jobs", type=int, default=10)
parser.add_argument("--dry_run", action='store_true')
parser.add_argument("--settings", type=str, action='append')


def estimate_runtime(setting: Dict):
  """Estimate a needed runtime in minutes."""
  data = setting["data"]
  N = data["N"]

  if N >= 1000:
    est_minutes = 20
  elif N >= 500:
    est_minutes = 10
  else:
    est_minutes = 5
  return est_minutes


PreparedSetting = collections.namedtuple(
    "PreparedSetting", "setting est_time config_file_path cmd")


def prepare_settings(settings: List[Dict], config_dir: str,
                     storage_dir: str) -> List[PreparedSetting]:
  """Preprocess the settings."""
  prepared_settings = []
  for setting in settings:
    setting_json = json.dumps(setting, indent=4, sort_keys=True)
    setting_md5 = hashlib.md5(setting_json.encode('utf-8')).hexdigest()
    config_file_path = os.path.join(config_dir, f"{setting_md5}.json")
    with open(config_file_path, "w") as config_file:
      config_file.write(setting_json)

    cmd = textwrap.dedent(f"""
        python -m distributed_cox.experiments.run -p -F {storage_dir} with \\
          {config_file_path} \\
          num_experiments=10000
        """)
    prepared_settings.append(
        PreparedSetting(setting, estimate_runtime(setting), config_file_path,
                        cmd))
  return prepared_settings


def partition_settings(settings: List[PreparedSetting],
                       n_partitions: int) -> List[List[PreparedSetting]]:
  groups = [[] for _ in range(n_partitions)]
  sums = np.zeros(n_partitions, dtype=int)
  settings = sorted(settings, key=lambda s: s.est_time)
  for setting in settings:
    min_idx = np.argmin(sums)
    sums[min_idx] += setting.est_time
    groups[min_idx].append(setting)
  return groups


def format_timedelta(value,
                     time_format="{days} days, {hours2}:{minutes2}:{seconds2}"):
  """Format a datetie.timedelta. See """
  if hasattr(value, 'seconds'):
    seconds = value.seconds + value.days * 24 * 3600
  else:
    seconds = int(value)

  seconds_total = seconds

  minutes = int(math.floor(seconds / 60))
  minutes_total = minutes
  seconds -= minutes * 60

  hours = int(math.floor(minutes / 60))
  hours_total = hours
  minutes -= hours * 60

  days = int(math.floor(hours / 24))
  days_total = days
  hours -= days * 24

  years = int(math.floor(days / 365))
  years_total = years
  days -= years * 365

  return time_format.format(
      **{
          'seconds': seconds,
          'seconds2': str(seconds).zfill(2),
          'minutes': minutes,
          'minutes2': str(minutes).zfill(2),
          'hours': hours,
          'hours2': str(hours).zfill(2),
          'days': days,
          'years': years,
          'seconds_total': seconds_total,
          'minutes_total': minutes_total,
          'hours_total': hours_total,
          'days_total': days_total,
          'years_total': years_total,
      })


fake_job_id = 0


def run_settings_on_node(slurm: simple_slurm.Slurm,
                         settings: List[PreparedSetting],
                         dry_run=True):
  """Run settings in a single sbatch command."""
  all_cmds = "\n\n".join([s.cmd for s in settings])
  est_sum_time = sum([s.est_time for s in settings])
  # Over-estimate our time a bit to be super safe
  est_duration = datetime.timedelta(minutes=est_sum_time * 1.2)

  slurm.add_arguments(time=format_timedelta(
      est_duration, "{days}-{hours2}:{minutes2}:{seconds2}"))
  cmd = textwrap.dedent(f"""
        ROOT_DIR={root_dir}
        source activate varderiv
        cd $ROOT_DIR

        {all_cmds}
        """)
  if not dry_run:
    return slurm.sbatch(cmd, shell="/bin/bash")
  global fake_job_id  # pylint: disable=global-statement
  fake_job_id += 1
  return fake_job_id


def run_settings(slurm: simple_slurm.Slurm,
                 settings: List[Dict],
                 config_dir: str,
                 storage_dir: str,
                 n_jobs=10,
                 dry_run=True):
  """Run the settings in groups."""
  settings = prepare_settings(settings, config_dir, storage_dir)
  groups = partition_settings(settings, n_jobs)
  job_id_to_config_paths = {}
  for group in groups:
    job_id = run_settings_on_node(slurm, group, dry_run=dry_run)
    job_id_to_config_paths[job_id] = {
        "est_time": sum([s.est_time for s in group]),
        "configs": [s.config_file_path for s in group]
    }

  with open(os.path.join(config_dir, "jobs.json"), "w") as f:
    json.dump(job_id_to_config_paths, f, indent=4, sort_keys=True)


def import_file(full_name, path):
  """Import a python module from a path. 3.4+ only.

    Does not call sys.modules[full_name] = path
  """

  spec = util.spec_from_file_location(full_name, path)
  mod = util.module_from_spec(spec)

  spec.loader.exec_module(mod)
  return mod


def main():
  args = parser.parse_args()
  if args.settings is None:
    args.settings = ["paper"]

  settings = []
  for i, settings_path in enumerate(args.settings):
    setting_mod = import_file(f"settings_{i}", settings_path)
    settings += setting_mod.settings

  slurm = simple_slurm.Slurm(cpus_per_task=8,
                             nodes=1,
                             time='0-1:00',
                             partition='short',
                             mem=3000,
                             output=os.path.join(args.logs_dir,
                                                 'hostname_%j.out'),
                             error=os.path.join(args.logs_dir,
                                                'hostname_%j.err'),
                             mail_type='FAIL',
                             mail_user='dl263@hms.harvard.edu')

  run_settings(slurm,
               settings,
               args.config_dir,
               args.storage_dir,
               args.n_jobs,
               dry_run=args.dry_run)


if __name__ == "__main__":
  main()
