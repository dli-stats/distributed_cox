"""Meta Analysis single experiment."""

import tempfile
import collections
import functools
import pickle

import numpy as onp

import jax.numpy as np
from jax import jit

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from varderiv.data import group_data_by_labels

from varderiv.equations.eq1 import solve_eq1_ad, solve_eq1_manual
from varderiv.equations.eq1 import eq1_compute_H_ad, eq1_compute_H_manual

from varderiv.equations.eq2 import solve_grouped_eq_batch
from varderiv.equations.meta_analysis import get_cov_meta_analysis_fn
from varderiv.equations.meta_analysis import get_meta_analysis_rest_solver

from varderiv.experiments.utils import expand_namedtuples
from varderiv.experiments.utils import run_cov_experiment
from varderiv.experiments.utils import CovExperimentResultItem
from varderiv.experiments.utils import check_value_converged

from varderiv.experiments.common import ingredient as base_ingredient

# pylint: disable=missing-function-docstring

ExperimentMetaAnalysisSolResult = collections.namedtuple(
    "ExperimentMetaAnalysisSolResult", "pt1 pt2")


def cov_experiment_meta_analysis_init(params):

  if params["eq1_cov_use_ad"]:
    eq1_compute_H_fn = eq1_compute_H_ad
  else:
    eq1_compute_H_fn = eq1_compute_H_manual
  del params["eq1_cov_use_ad"]

  if params["solve_eq1_use_ad"]:
    solve_eq1_fn = solve_eq1_ad
  else:
    solve_eq1_fn = solve_eq1_manual

  if params["slice_X_DIMs"] is not None and params["post_slice_X_DIMs"]:
    post_slice_X_DIMs = params["slice_X_DIMs"]
    params["slice_X_DIMs"] = None
  else:
    post_slice_X_DIMs = None
  del params["post_slice_X_DIMs"]

  solve_meta_analysis_fn = functools.partial(
      solve_grouped_eq_batch,
      solve_eq1_fn=jit(solve_eq1_fn),
      solve_rest_fn=jit(
          get_meta_analysis_rest_solver(eq1_compute_H_fn=eq1_compute_H_fn,
                                        slice_X_DIMs=post_slice_X_DIMs)))
  params["solve_meta_analysis_fn"] = solve_meta_analysis_fn
  del params["solve_eq1_use_ad"]

  params["meta_analysis_cov_fn"] = jit(
      get_cov_meta_analysis_fn(eq1_compute_H_fn=eq1_compute_H_fn,
                               slice_X_DIMs=post_slice_X_DIMs))


def cov_experiment_meta_analysis_core(rnd_keys,
                                      N=1000,
                                      X_DIM=4,
                                      K=3,
                                      slice_X_DIMs=None,
                                      gen=None,
                                      solve_meta_analysis_fn=None,
                                      meta_analysis_cov_fn=None):
  del N
  assert gen is not None
  assert solve_meta_analysis_fn is not None
  assert meta_analysis_cov_fn is not None

  key, data_generation_key = map(np.array, zip(*rnd_keys))
  del key  # Not used currently

  X, delta, beta, group_labels = gen(data_generation_key)

  if slice_X_DIMs is not None:
    # Performa Meta Analysis on only certain X_DIMs
    X = np.take(X, slice_X_DIMs, axis=-1)
    beta = np.take(beta, slice_X_DIMs, axis=-1)
    X_DIM = len(slice_X_DIMs)

  batch_size = len(X)
  assert beta.shape == (batch_size, X_DIM)

  X_groups, delta_groups = group_data_by_labels(batch_size, K, X, delta,
                                                group_labels)

  pt1_sols, pt2_sols = solve_meta_analysis_fn(X,
                                              delta,
                                              K,
                                              group_labels,
                                              X_groups=X_groups,
                                              delta_groups=delta_groups,
                                              initial_guess=beta,
                                              log=False)

  beta_k_hat = pt1_sols.guess
  beta_hat = pt2_sols.guess

  cov = onp.array(
      meta_analysis_cov_fn(X, delta, X_groups, delta_groups, group_labels,
                           beta_k_hat, beta_hat))

  pt1_sols = expand_namedtuples(type(pt1_sols)(*map(onp.array, pt1_sols)))
  pt2_sols = expand_namedtuples(type(pt2_sols)(*map(onp.array, pt2_sols)))

  ret = expand_namedtuples(
      CovExperimentResultItem(sol=expand_namedtuples(
          ExperimentMetaAnalysisSolResult(pt1=pt1_sols, pt2=pt2_sols)),
                              cov=cov))
  return ret


cov_experiment_meta_analysis = functools.partial(
    run_cov_experiment,
    cov_experiment_meta_analysis_init,
    cov_experiment_meta_analysis_core,
    check_fail_fn=lambda r: check_value_converged(r.sol.pt2.value))

ex = Experiment("meta_analysis", ingredients=[base_ingredient])


@ex.config
def config():
  # pylint: disable=unused-variable
  solve_eq1_use_ad = True
  eq1_cov_use_ad = True
  slice_X_DIMs = None
  post_slice_X_DIMs = False


@ex.main
def cov_experiment_meta_analysis_main(base, solve_eq1_use_ad, eq1_cov_use_ad):
  # pylint: disable=missing-function-docstring
  base = dict(base)
  base.pop("seed")
  pickle.dump(
      cov_experiment_meta_analysis(solve_eq1_use_ad=solve_eq1_use_ad,
                                   eq1_cov_use_ad=eq1_cov_use_ad,
                                   **base), result_file)
  ex.add_artifact(result_file.name, name="result")


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  result_file = tempfile.NamedTemporaryFile(mode="wb+")
  ex.run_commandline()
