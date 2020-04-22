"""Eq2 single experiment."""

import tempfile
import collections
import functools
import pickle

import numpy as onp

import jax.numpy as np
from jax import jit

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from varderiv.data import data_generator, group_labels_generator
from varderiv.data import group_data_by_labels

from varderiv.equations.eq1 import solve_eq1_ad, solve_eq1_manual
from varderiv.equations.eq1 import eq1_compute_H_ad, eq1_compute_H_manual

from varderiv.equations.eq2 import solve_grouped_eq_batch
from varderiv.equations.eq2 import get_eq2_cov_beta_k_correction_fn
from varderiv.equations.eq2 import eq2_solve_rest
from varderiv.equations.eq2 import eq2_cov_robust_ad_impl

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem
from varderiv.experiments.utils import check_value_converged

from varderiv.experiments.common import ingredient as base_ingredient
from varderiv.experiments.grouped_common import ingredient as grouped_ingredient

# pylint: disable=missing-function-docstring

Experiment2SolResult = collections.namedtuple("Experiment2SolResult", "pt1 pt2")

Experiment2CovResult = collections.namedtuple(
    "Experiment2CovResult", "cov_beta_k_correction cov_robust_ad cov_H")


def cov_experiment_eq2_init(params):
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

  if params["solve_eq1_use_ad"]:
    solve_eq1_fn = solve_eq1_ad
  else:
    solve_eq1_fn = solve_eq1_manual
  solve_eq2 = functools.partial(solve_grouped_eq_batch,
                                solve_eq1_fn=jit(solve_eq1_fn),
                                solve_rest_fn=jit(eq2_solve_rest))
  params["solve_eq2_fn"] = solve_eq2
  del params["solve_eq1_use_ad"]

  if params["eq1_cov_use_ad"]:
    eq1_compute_H_fn = eq1_compute_H_ad
  else:
    eq1_compute_H_fn = eq1_compute_H_manual
  del params["eq1_cov_use_ad"]

  params["cov_beta_k_correction_fn"] = jit(
      get_eq2_cov_beta_k_correction_fn(eq1_compute_H_fn=eq1_compute_H_fn))
  params["cov_robust_fn"] = jit(eq2_cov_robust_ad_impl)


def cov_experiment_eq2_core(rnd_keys,
                            N=1000,
                            X_DIM=4,
                            K=3,
                            gen=None,
                            group_labels_gen=None,
                            solve_eq2_fn=None,
                            cov_beta_k_correction_fn=None,
                            cov_robust_fn=None):
  del N
  assert gen is not None
  assert group_labels_gen is not None
  assert solve_eq2_fn is not None
  assert cov_beta_k_correction_fn is not None
  assert cov_robust_fn is not None

  key, data_generation_key = map(np.array, zip(*rnd_keys))

  X, delta, beta = gen(data_generation_key)
  group_labels = group_labels_gen(data_generation_key)

  batch_size = len(X)
  assert beta.shape == (batch_size, X_DIM)

  X_groups, delta_groups = group_data_by_labels(batch_size, K, X, delta,
                                                group_labels)

  pt1_sols, pt2_sols = solve_eq2_fn(X,
                                    delta,
                                    K,
                                    group_labels,
                                    X_groups=X_groups,
                                    delta_groups=delta_groups,
                                    initial_guess=beta,
                                    log=False)

  beta_k_hat = pt1_sols.guess
  beta_hat = pt2_sols.guess

  cov_beta_k_correction = cov_beta_k_correction_fn(X, delta, X_groups,
                                                   delta_groups, group_labels,
                                                   beta_k_hat, beta_hat)
  cov_beta_k_correction = onp.array(cov_beta_k_correction)

  cov_H, cov_robust_ad = cov_robust_fn(X, delta, group_labels, beta_k_hat, beta)
  cov_H = onp.array(cov_H)
  cov_robust_ad = onp.array(cov_robust_ad)

  pt1_sols = expand_namedtuples(type(pt1_sols)(*map(onp.array, pt1_sols)))
  pt2_sols = expand_namedtuples(type(pt2_sols)(*map(onp.array, pt2_sols)))

  ret = expand_namedtuples(
      CovExperimentResultItem(
          sol=expand_namedtuples(
              Experiment2SolResult(pt1=pt1_sols, pt2=pt2_sols)),
          cov=expand_namedtuples(
              Experiment2CovResult(cov_beta_k_correction=cov_beta_k_correction,
                                   cov_robust_ad=cov_robust_ad,
                                   cov_H=cov_H))))
  return ret


cov_experiment_eq2 = functools.partial(
    run_cov_experiment,
    cov_experiment_eq2_init,
    cov_experiment_eq2_core,
    check_fail_fn=lambda r: check_value_converged(r.sol.pt2.value))

ex = Experiment("eq2", ingredients=[base_ingredient, grouped_ingredient])


@ex.config
def config():
  # pylint: disable=unused-variable
  solve_eq1_use_ad = True
  eq1_cov_use_ad = True


@ex.main
def cov_experiment_eq2_main(base, grouped, solve_eq1_use_ad, eq1_cov_use_ad):
  # pylint: disable=missing-function-docstring
  base = dict(base)
  base.pop("seed")
  pickle.dump(
      cov_experiment_eq2(solve_eq1_use_ad=solve_eq1_use_ad,
                         eq1_cov_use_ad=eq1_cov_use_ad,
                         **base,
                         **grouped), result_file)
  ex.add_artifact(result_file.name, name="result")


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  result_file = tempfile.NamedTemporaryFile(mode="wb+")
  ex.run_commandline()
