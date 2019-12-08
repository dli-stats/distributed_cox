"""Distributed Ver. Commandline tool.

Usage:
    cmd.py dummy-gen <N> <X_DIM> <K> [options]
    cmd.py eq2 local-send-T <DATA_FILE> [options]
    cmd.py eq2 master-send-T <DATA_DIR> [options]
    cmd.py eq2 local-send-vals <T_DELTA_FILE> <DATA_FILE> [options]
    cmd.py eq4 local-send-vals <DATA_FILE> [options]
    cmd.py <EQ> master-analytics <DATA_DIR> [options]

Options:
  -h --help               Show this screen.
  --version               Show version.
  --key=<key>             Use integer random key.  [default: 0]
  --save-dir=<save_dir>   Directory to save. [default: ./]

"""

import os
import glob

from docopt import docopt

import numpy as onp

import jax.random as jrandom

from varderiv.data import data_generator
from varderiv.data import group_labels_generator

from varderiv.distributed.eq2 import distributed_compute_eq2_local
from varderiv.distributed.eq2 import distributed_compute_eq2_master

from varderiv.distributed.eq4 import distributed_compute_eq4_local
from varderiv.distributed.eq4 import distributed_compute_eq4_master


def run_dummy_gen(args):
  N = int(args['<N>'])
  X_DIM = int(args['<X_DIM>'])
  K = int(args['<K>'])
  k1, k2 = jrandom.split(args['--key'])
  T, X, delta, _ = data_generator(N, X_DIM, return_T=True)(k1)
  group_labels = group_labels_generator(N, K, "same")(k2)

  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    onp.savez(os.path.join(args['--save-dir'], "local_" + str(k)),
              X_group=X_group,
              T_group=T_group,
              delta_group=delta_group)


def run_local_send_T(args):
  npzfile = onp.load(args['<DATA_FILE>'])
  T_group = npzfile['T_group']
  delta_group = npzfile['delta_group']
  T_delta_group = T_group[delta_group == 1]
  onp.savez(os.path.join(
      args['--save-dir'],
      "T_delta_group_" + os.path.basename(args['<DATA_FILE>'])),
            T_delta_group=T_delta_group)


def run_master_send_T(args):
  delta_T_group_files = glob.glob(
      os.path.join(args['<DATA_DIR>'], "T_delta_group_*"))
  T_delta = [onp.load(dtgf)['T_delta_group'] for dtgf in delta_T_group_files]
  T_delta = onp.concatenate(T_delta)
  T_delta = onp.sort(T_delta)
  onp.savez(os.path.join(args["--save-dir"], "T_delta"), T_delta=T_delta)


def run_local_send_vals(args):
  distributed_compute_eqn_fn_map = {
      "eq2": distributed_compute_eq2_local,
      "eq4": distributed_compute_eq4_local
  }
  npzfile = onp.load(args['<DATA_FILE>'])
  T_group, X_group, delta_group = npzfile['T_group'], npzfile[
      'X_group'], npzfile['delta_group']
  if args['eq2']:
    T_delta = onp.load(args['<T_DELTA_FILE>'])['T_delta']
    local_vals = distributed_compute_eq2_local(args['--key'], T_group, X_group,
                                               delta_group, T_delta)
  elif args['eq4']:
    local_vals = distributed_compute_eq4_local(args['--key'], T_group, X_group,
                                               delta_group)
  else:
    raise TypeError("Invalid command")

  onp.savez(
      os.path.join(args["--save-dir"],
                   "local_vals_" + os.path.basename(args['<DATA_FILE>'])),
      *local_vals)


def run_master_analytics(args):
  local_val_files = glob.glob(os.path.join(args['<DATA_DIR>'], "local_vals_*"))
  local_vals = [onp.load(lvf) for lvf in local_val_files]
  local_vals = [tuple(lv[lvf] for lvf in lv.files) for lv in local_vals]

  def list_or_array(data):
    shape = data[0].shape
    for d in data:
      if not d.shape == shape:
        return list(data)
    return onp.stack(data)

  local_vals = tuple(
      list_or_array([ld[d]
                     for ld in local_vals])
      for d, _ in enumerate(local_vals[0]))

  if args['<EQ>'] == 'eq2':
    distributed_compute_eq_master = distributed_compute_eq2_master
  elif args['<EQ>'] == 'eq4':
    distributed_compute_eq_master = distributed_compute_eq4_master
  else:
    raise TypeError("Invalid command")

  beta_hat, beta_corrected_cov = distributed_compute_eq_master(*local_vals)
  print("beta:", beta_hat)
  print("beta_cov:", beta_corrected_cov)


def main():
  args = docopt(__doc__)
  args['--key'] = jrandom.PRNGKey(int(args['--key']))
  if args['dummy-gen']:
    run_dummy_gen(args)
  elif args['local-send-T']:
    run_local_send_T(args)
  elif args['master-send-T']:
    run_master_send_T(args)
  elif args['local-send-vals']:
    run_local_send_vals(args)
  elif args['master-analytics']:
    run_master_analytics(args)


if __name__ == "__main__":
  main()
