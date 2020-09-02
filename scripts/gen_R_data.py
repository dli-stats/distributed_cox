"""Generate R data."""

import argparse
import os

import numpy as onp
import jax.random as jrandom

import distributed_cox.data as data
import distributed_cox.experiments.common as experiment_common
import distributed_cox.experiments.utils as experiment_utils

# pylint: disable=missing-function-docstring


def gen_R_data(outdir,
               data_generation_key=data.data_generation_key,
               num_experiments=10000,
               **experiment_params):
  experiment_params = experiment_common.process_params(**experiment_params)
  data_generation_subkeys = jrandom.split(data_generation_key, num_experiments)
  experiment_utils.init_data_gen_fn(experiment_params,
                                    return_T=True,
                                    return_T_star=True)
  data_gen = experiment_params["gen"]
  T_star, T, X, delta, beta, group_labels = data_gen(data_generation_subkeys)  # pylint: disable=unused-variable
  os.makedirs(outdir, exist_ok=True)
  for vname in ["T_star", "T", "X", "delta", "beta", "group_labels"]:
    a = eval(vname)  # pylint: disable=eval-used
    a = a.reshape((-1, a.shape[-1]))
    onp.savetxt(os.path.join(outdir, vname), a, delimiter=",")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("outdir", type=str, default="out/")
  parser.add_argument("--N", type=int)
  parser.add_argument("--X_DIM", type=int)
  parser.add_argument("--K", type=int)
  parser.add_argument("--group_labels_generator_kind",
                      choices=['same', 'arithmetic_sequence'])
  parser.add_argument("--group_X_same", type=bool)
  parser.add_argument("--T_star_factors",
                      type=str,
                      choices=['None', 'fixed', 'gamma'])
  args = parser.parse_args()
  if args.T_star_factors == "None":
    args.T_star_factors = None
  gen_R_data(args.outdir,
             N=args.N,
             X_DIM=args.X_DIM,
             K=args.K,
             group_labels_generator_kind=args.group_labels_generator_kind,
             group_X_same=args.group_X_same,
             T_star_factors=args.T_star_factors)
