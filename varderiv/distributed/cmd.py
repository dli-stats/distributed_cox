"""Distributed Ver. Commandline tool.

Usage:
    cmd.py dummy-gen <N> <X_DIM> <K> [-h -k=<key> --save-dir=<save_dir> --save-global]
    cmd.py eq2 local-send-T <DATA_FILE> [-h -k=<key> --save-dir=<save_dir>]
    cmd.py eq2 master-send-T <DATA_DIR> [-h -k=<key> --save-dir=<save_dir>]
    cmd.py eq2 local-send-vals <T_DELTA_FILE> <DATA_FILE> [-h -k=<key> --save-dir=<save_dir> --solver-thres=<thres> --solver-max-steps=<N>]
    cmd.py eq4 local-send-vals <DATA_FILE> [-h -k=<key> --save-dir=<save_dir> --solver-thres=<thres> --solver-max-steps=<N>]
    cmd.py (eq2 | eq4) master-analytics <DATA_DIR> [-h -k=<key> --solver-thres=<thres> --solver-max-steps=<N> --save-dir=<save_dir>]
    cmd.py (eq1 | eq3) master-analytics <DATA_FILE> [-h -k=<key> --solver-thres=<thres> --solver-max-steps=<N> --save-dir=<save_dir>]

Options:
  -h --help                  Show this screen.
  -k --key=<key>             Use integer random key.  [default: 0]
  --save-dir=<save_dir>      Directory to save.
  --save-global              Save global generated data.
  --solver-thres=<thres>     Solver threshold. [default: 1e-3]
  --solver-max-steps=<N>     Solver maximum steps. [default: 10]
"""

import os
import glob

from docopt import docopt

import numpy as onp
import pandas as pd

import jax.random as jrandom

from varderiv.data import data_generator
from varderiv.data import group_labels_generator
from varderiv.data import group_data_by_labels

from varderiv.equations.eq1 import get_eq1_solver, eq1_cov, eq1_log_likelihood_grad_ad
from varderiv.equations.eq3 import get_eq3_solver, get_eq3_cov_fn

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
  if args['save_global']:
    onp.savez(os.path.join(args['--save-dir'], 'all'), 
      X=X,
      T=T,
      delta=delta,
      group_labels=group_labels
    )

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
  solve_eq1_fn = get_eq1_solver(use_ad=True, 
    solver_max_steps=int(args['--solver-max-steps']), 
    norm_stop_thres=float(args['--solver-thres']))
  distributed_compute_eqn_fn_map = {
      "eq2": distributed_compute_eq2_local,
      "eq4": distributed_compute_eq4_local
  }
  npzfile = onp.load(args['<DATA_FILE>'])
  T_group, X_group, delta_group = (npzfile['T_group'], npzfile['X_group'], npzfile['delta_group'])
  # TODO randomized guess
  if args['eq2']:
    T_delta = onp.load(args['<T_DELTA_FILE>'])['T_delta']
    local_vals = distributed_compute_eq2_local(T_group, X_group,
                                               delta_group, T_delta, 
                                               solve_eq1_fn=solve_eq1_fn)
  elif args['eq4']:
    local_vals = distributed_compute_eq4_local(T_group, X_group,
                                               delta_group, 
                                               solve_eq1_fn=solve_eq1_fn)
  else:
    raise TypeError("Invalid command")
  print(local_vals)
  onp.savez(
      os.path.join(args["--save-dir"],
                   "local_vals_" + os.path.basename(args['<DATA_FILE>'])),
      *local_vals)


def run_master_analytics(args):
  solver_max_steps, solver_thres = int(args['--solver-max-steps']), float(args['--solver-thres'])
  eq = next(k for k in {'eq1', 'eq2', 'eq3', 'eq4'} if args[k])
  if eq in ('eq1', 'eq3'):
    global_data = onp.load(args['<DATA_FILE>'])
    X, delta = global_data['X'], global_data['delta']
    beta_guess = onp.zeros(X.shape[1])
    if eq == 'eq1':
      solve_eq1 = get_eq1_solver(use_ad=True, 
        solver_max_steps=solver_max_steps, norm_stop_thres=solver_thres)
      sol = solve_eq1(X, delta, beta_guess)
      beta_hat = sol.guess
      beta_cov = eq1_cov(X, delta, beta_hat, use_ad=True)

    elif eq == 'eq3':
      eq1_ll_grad_fn = eq1_log_likelihood_grad_ad
      group_labels = global_data['group_labels']
      K = group_labels.max() + 1
      X_groups, delta_groups = group_data_by_labels(1, K, X, delta, group_labels)
      solve_eq3 = get_eq3_solver(eq1_ll_grad_fn, 
        solver_max_steps=solver_max_steps, norm_stop_thres=solver_thres)
      sol = solve_eq3(X_groups, delta_groups, beta_guess)
      import pdb; pdb.set_trace()
      beta_hat = sol.guess
      eq3_cov_fn = get_eq3_cov_fn(eq1_ll_grad_fn)
      beta_cov = eq3_cov_fn(X_groups, delta_groups, beta_hat)
    
  elif eq in ('eq2', 'eq4'):
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

    if eq == 'eq2':
      distributed_compute_eq_master = distributed_compute_eq2_master
    elif eq == 'eq4':
      distributed_compute_eq_master = distributed_compute_eq4_master

    beta_hat, beta_cov = distributed_compute_eq_master(*local_vals)
  
  print("beta:", beta_hat)
  print("beta variance:", beta_cov.diagonal())

  save_dir = args['--save-dir']
  pd.DataFrame(beta_hat).to_csv(os.path.join(save_dir, "result_{eq}_beta_hat.csv".format(eq=eq)))
  pd.DataFrame(beta_cov).to_csv(os.path.join(save_dir, "result_{eq}_beta_cov.csv".format(eq=eq)))


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
