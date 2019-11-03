"""Eq3 single experiment."""

import functools
import pickle

import numpy as onp

import jax.numpy as np
from jax import jit

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from varderiv.data import data_generator, group_labels_generator
from varderiv.data import group_data_by_labels, group_labels_to_indices

from varderiv.equations.eq1 import eq1_log_likelihood_grad_ad
from varderiv.equations.eq1 import eq1_log_likelihood_grad_manual

from varderiv.equations.eq3 import eq3_solver, get_eq3_cov_fn

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem

from varderiv.experiments.common import ingredient as base_ingredient
from varderiv.experiments.grouped_common import ingredient as grouped_ingredient

# pylint: disable=missing-function-docstring


def cov_experiment_eq3_init(params):
  params["group_labels_gen"] = jit(
      group_labels_generator(
          params["N"],
          params["K"],
          group_labels_generator_kind=params["group_labels_generator_kind"],
          **params["group_labels_generator_kind_kwargs"]))
  del params["group_labels_generator_kind"]
  del params["group_labels_generator_kind_kwargs"]

  gen = jit(data_generator(params["N"], params["X_DIM"]))
  params["gen"] = gen

  if params["eq1_ll_grad_use_ad"]:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_ad
  else:
    eq1_ll_grad_fn = eq1_log_likelihood_grad_manual
  solve_eq3_fn = jit(eq3_solver(eq1_ll_grad_fn))
  eq3_cov_fn = jit(get_eq3_cov_fn(eq1_ll_grad_fn))

  params["solve_eq3_fn"] = solve_eq3_fn
  params["eq3_cov_fn"] = eq3_cov_fn
  del params["eq1_ll_grad_use_ad"]


def cov_experiment_eq3_core(rnd_keys,
                            N=1000,
                            X_DIM=4,
                            K=3,
                            gen=None,
                            group_labels_gen=None,
                            solve_eq3_fn=None,
                            eq3_cov_fn=None):
  del N, X_DIM
  assert gen is not None
  assert group_labels_gen is not None
  assert solve_eq3_fn is not None
  assert eq3_cov_fn is not None

  key, data_generation_key = map(np.array, zip(*rnd_keys))

  X, delta, beta = gen(data_generation_key)
  group_labels = group_labels_gen(data_generation_key)

  batch_size = len(X)

  group_indices = group_labels_to_indices(K, group_labels)
  X_groups, delta_groups = group_data_by_labels(batch_size, K, X, delta,
                                                group_indices)

  sol = solve_eq3_fn(key, X_groups, delta_groups, beta)
  beta_hat = sol.guess

  sol = expand_namedtuples(type(sol)(*map(onp.array, sol)))

  cov = onp.array(eq3_cov_fn(X_groups, delta_groups, beta_hat))

  ret = expand_namedtuples(CovExperimentResultItem(sol=sol, cov=cov))
  return ret


cov_experiment_eq3 = functools.partial(run_cov_experiment,
                                       cov_experiment_eq3_init,
                                       cov_experiment_eq3_core)

ex = Experiment("eq3", ingredients=[base_ingredient, grouped_ingredient])


@ex.config
def config():
  # pylint: disable=unused-variable
  eq1_ll_grad_use_ad = True


@ex.main
def cov_experiment_eq3_main(base, grouped, eq1_ll_grad_use_ad):
  # pylint: disable=missing-function-docstring
  base = dict(base)
  base.pop("seed")
  with open("result.pkl", "wb+") as f:
    pickle.dump(
        cov_experiment_eq3(eq1_ll_grad_use_ad=eq1_ll_grad_use_ad,
                           **base,
                           **grouped), f)
    ex.add_artifact("result.pkl")


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  ex.run_commandline()
