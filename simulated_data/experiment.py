import os

import argparse
import numpy as onp
import pandas as pd
import jax
from jax import jit
import jax.numpy as np

import distributed_cox.equations.eq1 as eq1
import distributed_cox.data


def data_from_csv(csv_file: str):
  dataframe = pd.read_csv(csv_file)
  x_headers = ["A"] + list(
      sorted((k for k in dataframe.keys() if k.startswith("X") and len(k) > 1),
             key=lambda s: int(s[1:])))
  X = dataframe[x_headers]
  delta = dataframe["status"]
  T = dataframe["time"]
  group_labels = dataframe["indDP"] - 1
  X, delta, T, group_labels = map(lambda x: x.to_numpy(),
                                  (X, delta, T, group_labels))

  sorted_idx = onp.argsort(-T)
  T = onp.take(T, sorted_idx, axis=0)
  X = onp.take(X, sorted_idx, axis=0)
  delta = onp.take(delta, sorted_idx, axis=0)
  group_labels = onp.take(group_labels, sorted_idx, axis=0)

  return X, delta, group_labels, T


def convert_from_csv(csv_file: str, save_dir: str):
  X, delta, group_labels, T = data_from_csv(csv_file)
  onp.savez(os.path.join(save_dir, "all"),
            X=X,
            T=T,
            delta=delta,
            group_labels=group_labels)
  K = group_labels.max() + 1
  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    onp.savez(os.path.join(save_dir, "local_" + str(k)),
              X_group=X_group,
              T_group=T_group,
              delta_group=delta_group)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-c', '--convert_data', action='store_true')
  parser.add_argument('-d',
                      '--data-dir',
                      type=str,
                      default='../distributed_multivariablecox_data/')
  parser.add_argument('-o', '--out-dir', type=str, default='analysis')
  parser.add_argument('-i', '--in-csv', type=str, default='dat.csv')
  args = parser.parse_args()

  csv_path = os.path.join(args.data_dir, args.in_csv)
  convert_data = True
  if convert_data:
    analysis_dir = os.path.join(args.data_dir, args.out_dir)
    os.makedirs(analysis_dir, exist_ok=True)
    convert_from_csv(csv_path, analysis_dir)
  else:
    X, delta, group_labels, T = data_from_csv(csv_path)
    eq1_solver = jit(
        eq1.get_eq1_solver(use_ad=True,
                           solver_max_steps=40,
                           norm_stop_thres=1e-1))
    beta_guess = onp.random.normal(0, 0.1, (X.shape[1],))
    print("Guess:", beta_guess)
    beta_correct = onp.array([
        -0.35044839, 0.02395054, -0.01438011, 0.10463269, -0.19862842,
        -0.23871021, -0.04685478, 0.10718658, 0.32139027, 0.14003561,
        0.34813819, -0.24538269, -0.07905443, 0.02583216, 0.09991671
    ])

    X_normalized, initial_beta_normalized, scale = varderiv.data.normalize(
        X, beta_guess)
    sol = eq1_solver(X_normalized, delta, initial_beta_normalized)
