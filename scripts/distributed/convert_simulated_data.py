from typing import Set

import pathlib
import re

import pandas as pd
import numpy as np

from distributed_cox import simple_cmd, cox_solve

cmd = simple_cmd.SimpleCMD()


def _x_keys_in_df(df: pd.DataFrame):
  ret = list(
      sorted((k for k in df.keys() if re.match(r'X\d+', k) and len(k) > 1),
             key=lambda s: int(s[1:])))
  if "A" in df.keys():
    ret = ["A"] + ret
  return ret


def df_to_numpy(df: pd.DataFrame):
  """Converts data in dataframe format to numpy format."""
  x_headers = _x_keys_in_df(df)
  X = df[x_headers]
  delta = df["status"]
  T = df["time"]
  group_labels = df["indDP"] - 1
  X, delta, T, group_labels = map(lambda x: x.to_numpy(),
                                  (X, delta, T, group_labels))
  delta = delta.astype(bool)
  sorted_idx = np.argsort(-T)
  T = np.take(T, sorted_idx, axis=0)
  X = np.take(X, sorted_idx, axis=0)
  delta = np.take(delta, sorted_idx, axis=0)
  group_labels = np.take(group_labels, sorted_idx, axis=0)

  return X, delta, group_labels, T, x_headers


def numpy_to_df(X, delta, group_labels, T, x_headers):
  """Converts data in numpy format to dataframe format."""
  data = {
      "status": delta.astype(int),
      "time": T,
      "indDP": group_labels + 1,
  }
  for i, xh in enumerate(x_headers):
    data[xh] = X[:, i]
  df = pd.DataFrame(data, columns=x_headers + ["status", "time", "indDP"])
  return df


def _regroup(X, delta, beta, K: int = 3, method: str = "even"):
  """Split X into groups."""
  sort_keys = np.dot(X, beta)
  N, _ = X.shape
  if method == "even":
    idxs = np.argsort(np.argsort(sort_keys))
    bins = [0] + [(N // K) * (i + 1) + min(i, N % K) for i in range(K)]
    print(bins)
    group_labels = np.digitize(idxs, np.array(bins)) - 1
    print(list(sort_keys[:20]))
    print(list(idxs[:20]))
    print(list(group_labels[:20]))
  elif method == "delta_even":
    group_labels = np.zeros(N, dtype=int)
    n_marked = np.sum(delta)
    print(n_marked)
    i = 0
    gl = 0
    for sk in list(np.argsort(sort_keys)):
      group_labels[sk] = gl
      if i >= n_marked // K and gl < K - 1:
        gl += 1
        i = 0
      if delta[sk]:
        i += 1
    print(max(group_labels))

  # print(N)
  # print(bins)
  # print(set(group_labels))
  print([np.sum(group_labels == k) for k in range(K)])
  return group_labels


@cmd.command
def convert_from_csv(csv_file: pathlib.Path, save_dir: pathlib.Path):
  df = pd.read_csv(csv_file)
  X, delta, group_labels, T, x_headers = df_to_numpy(df)
  save_dir.mkdir(exist_ok=True)
  np.savez(save_dir.joinpath("all"),
           X=X,
           T=T,
           delta=delta,
           group_labels=group_labels)
  K = group_labels.max() + 1
  for k in range(K):
    X_group = X[group_labels == k]
    T_group = T[group_labels == k]
    delta_group = delta[group_labels == k]
    np.savez(save_dir.joinpath(f"local_{k}"),
             X_group=X_group,
             T_group=T_group,
             delta_group=delta_group)


def _drop_columns(df: pd.DataFrame, exclude_columns: Set[str]):
  exclude_columns = exclude_columns - {"status", "time", "indDP"}
  return df.drop(columns=exclude_columns)


@cmd.command
def drop_columns(csv_file: pathlib.Path, save_csv_file: pathlib.Path,
                 columns: str):
  if columns == "":
    columns = set()
  else:
    columns = set(columns.split(","))
  df = pd.read_csv(csv_file)
  df = _drop_columns(df, columns)
  df.to_csv(save_csv_file, index=False)


@cmd.command
def regroup(csv_file: pathlib.Path,
            save_csv_file: pathlib.Path,
            K: int,
            exclude_columns: str,
            method: str = "delta_even"):
  df = pd.read_csv(csv_file)
  sorted_idx = np.argsort(-df["time"])
  unsort_idx = np.argsort(sorted_idx)

  df_subs = _drop_columns(df, set(exclude_columns.split(",")))
  X, delta, group_labels, T, _ = df_to_numpy(df)
  X_subs, delta, group_labels, T, _ = df_to_numpy(df_subs)

  cox_solve_fn = cox_solve.get_cox_solve_fn("unstratified_pooled")
  sol = cox_solve_fn(X_subs, delta, np.zeros(X_subs.shape[1]))

  new_group_labels = _regroup(X_subs, delta, sol.guess, K=K, method=method)
  df["indDP"] = pd.Series((new_group_labels + 1)[unsort_idx])
  df.to_csv(save_csv_file, index=False)


if __name__ == "__main__":
  cmd.run()
