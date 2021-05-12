import pathlib

import pandas as pd
import numpy as onp
import re

import distributed_cox.simple_cmd as simple_cmd

cmd = simple_cmd.SimpleCMD()


def data_from_csv(csv_file: pathlib.Path):
  dataframe = pd.read_csv(csv_file)
  x_headers = ["A"] + list(
      sorted(
          (k for k in dataframe.keys() if re.match(r'X\d+', k) and len(k) > 1),
          key=lambda s: int(s[1:])))
  X = dataframe[x_headers]
  delta = dataframe["status"]
  T = dataframe["time"]
  group_labels = dataframe["indDP"] - 1
  X, delta, T, group_labels = map(lambda x: x.to_numpy(),
                                  (X, delta, T, group_labels))
  delta = delta.astype(bool)
  sorted_idx = onp.argsort(-T)
  T = onp.take(T, sorted_idx, axis=0)
  X = onp.take(X, sorted_idx, axis=0)
  delta = onp.take(delta, sorted_idx, axis=0)
  group_labels = onp.take(group_labels, sorted_idx, axis=0)

  return X, delta, group_labels, T


@cmd.command
def convert_from_csv(csv_file: pathlib.Path, save_dir: pathlib.Path):
  X, delta, group_labels, T = data_from_csv(csv_file)
  onp.savez(save_dir.joinpath("all"),
            X=X,
            T=T,
            delta=delta,
            group_labels=group_labels)
  K = group_labels.max() + 1
  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    onp.savez(save_dir.joinpath(f"local_{k}"),
              X_group=X_group,
              T_group=T_group,
              delta_group=delta_group)


if __name__ == "__main__":
  cmd.run()
