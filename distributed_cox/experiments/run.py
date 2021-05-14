"""Basic Common configs for experiments."""

from typing import Dict, Optional, Union

import collections
import functools
import tempfile
import dataclasses
import os
import json

from frozendict import frozendict

import numpy as onp

from jax import jit, vmap
import jax.random as jrandom
import jax.tree_util as tu
import jax.numpy as jnp

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import distributed_cox.data as vdata

import distributed_cox.generic.modeling as modeling
from distributed_cox.generic.solver import NewtonSolverResult
import distributed_cox.cox_solve as cox_solve

import distributed_cox.experiments.utils as utils

# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring

ex = Experiment("cox")


@tu.register_pytree_node_class
@dataclasses.dataclass
class ExperimentResult:
  """For holding experiment results."""
  data_generation_key: onp.ndarray
  pt1: Optional[NewtonSolverResult]
  pt2: Union[NewtonSolverResult, modeling.MetaAnalysisResult]
  covs: Dict[str, onp.ndarray]

  def tree_flatten(self):
    if self.is_groupped:  # pylint: disable=no-else-return
      return (self.data_generation_key, self.pt1, self.pt2, self.covs), True
    else:
      return (self.data_generation_key, self.pt2, self.covs), False

  @classmethod
  def tree_unflatten(cls, aux, children):
    is_groupped = aux
    if is_groupped:  # pylint: disable=no-else-return
      return cls(*children)
    else:
      data_generation_key, pt2, covs = children
      return cls(data_generation_key, None, pt2, covs)

  @property
  def is_groupped(self):
    return self.pt1 is not None

  @property
  def sol(self):
    return self.pt2

  @property
  def guess(self):
    return self.sol.guess


def compute_results_averaged(result: ExperimentResult,
                             std=False,
                             keep_idxs=None):
  """
  Args:
    - result: the experiment result object.
    - A function given a result.sol, returns the beta
    - A dict from str to function. The string represents a key
      to plot the analytical cov. The function returns the cov
      matrices from result.cov.
  """
  if keep_idxs is None:
    keep_idxs = onp.ones_like(result.sol.converged, dtype=jnp.bool_)

  all_covs = collections.OrderedDict()

  converged_idxs = result.sol.converged
  if result.is_groupped:
    pt1_converged_idxs = onp.all(result.pt1.converged, axis=1)
    converged_idxs &= pt1_converged_idxs

  beta = result.sol.guess

  beta_empirical_nan_idxs = onp.any(onp.isnan(beta), axis=1)

  keep_idxs = keep_idxs & converged_idxs & ~beta_empirical_nan_idxs
  for cov_name, cov_analyticals in result.covs.items():
    cov_analyticals_nan_idxs = onp.any(onp.isnan(
        cov_analyticals.reshape(-1, cov_analyticals.shape[1]**2)),
                                       axis=1)
    keep_idxs &= ~cov_analyticals_nan_idxs

  beta = beta[keep_idxs]
  beta_hat = onp.average(beta, axis=0)

  cov_empirical = onp.cov(beta, rowvar=False)
  if cov_empirical.shape == tuple():
    cov_empirical = cov_empirical.reshape((1, 1))
  if std:
    cov_empirical = onp.sqrt(jnp.diagonal(cov_empirical, axis1=-2, axis2=-1))
  all_covs["empirical"] = cov_empirical

  for cov_name, cov_analyticals in result.covs.items():
    cov_analyticals = cov_analyticals[keep_idxs]
    if std:
      cov_analyticals = jnp.sqrt(
          jnp.diagonal(cov_analyticals, axis1=-2, axis2=-1))
    cov_analytical = onp.mean(cov_analyticals, axis=0)
    all_covs[cov_name] = cov_analytical

  return beta_hat, all_covs, keep_idxs


@ex.capture(prefix="data")
def init_data_gen_fn(N,
                     K,
                     X_DIM,
                     T_star_factors,
                     group_labels_generator_kind,
                     group_X,
                     exp_scale,
                     npz_path: str = None):
  if npz_path is None:
    return vdata.full_data_generator(N, K, X_DIM, T_star_factors,
                                     group_labels_generator_kind, group_X,
                                     exp_scale)
  data = onp.load(npz_path)
  T, X, delta, group_labels = (data["T"], data["X"], data["delta"],
                               data["group_labels"])
  T_star = onp.zeros_like(T) - 1  # Not applicable for real data
  group_sizes = tuple(onp.sum(group_labels == k) for k in range(K))

  def gen(key):
    del key
    return T, T_star, X, delta, onp.zeros(X.shape[1]), group_labels

  return group_sizes, gen


def freezeargs(func):
  """
  Transform mutable dictionnary
  Into immutable
  Useful to be compatible with cache
  """

  @functools.wraps(func)
  def wrapped(*args, **kwargs):
    args = tuple(
        [frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
    kwargs = {
        k: frozendict(v) if isinstance(v, dict) else v
        for k, v in kwargs.items()
    }
    return func(*args, **kwargs)

  return wrapped


@ex.capture
@freezeargs
@functools.lru_cache(maxsize=None)
def cov_experiment_init(eq, distributed, solver, meta_analysis,
                        **experiment_params):
  del experiment_params

  group_sizes, gen = init_data_gen_fn()  # pylint: disable=no-value-for-parameter
  solve_and_cov_fn = cox_solve.get_cox_solve_and_cov_fn(eq, group_sizes,
                                                        distributed, solver,
                                                        meta_analysis)

  def solve_and_cov_end2end(data_generation_key):
    T_star, T, X, delta, beta, group_labels = gen(data_generation_key)
    del T_star, T  # not used by solver
    cox_sol = solve_and_cov_fn(X, delta, beta, group_labels)
    return ExperimentResult(data_generation_key=data_generation_key,
                            pt1=cox_sol.pt1,
                            pt2=cox_sol.pt2,
                            covs=cox_sol.covs)

  return {
      "solve_and_cov": jit(vmap(solve_and_cov_end2end)),
      "gen": jit(vmap(gen))
  }


class EndExperimentException(Exception):
  pass


@ex.capture
def cov_experiment_core(rnd_keys,
                        solve_and_cov=None,
                        gen=None,
                        save_data_csv=None,
                        data=None):
  assert solve_and_cov is not None
  key, data_generation_keys = map(jnp.array, zip(*rnd_keys))
  if save_data_csv is not None:
    T_star, T, X, delta, beta, group_labels = gen(data_generation_keys)
    os.makedirs(save_data_csv, exist_ok=True)
    for vname in ["T_star", "T", "X", "delta", "beta", "group_labels"]:
      a = eval(vname)  # pylint: disable=eval-used
      a = a.reshape((-1, a.shape[-1]))
      onp.savetxt(os.path.join(save_data_csv, vname), a, delimiter=",")
    with open(os.path.join(save_data_csv, "data.json"), "w+") as f:
      json.dump(data, f)
    raise EndExperimentException()
  del gen
  experiment_result = solve_and_cov(data_generation_keys)
  return experiment_result


cov_experiment = functools.partial(utils.run_cov_experiment,
                                   cov_experiment_init,
                                   cov_experiment_core,
                                   check_ok_fn=lambda r: r.sol.converged)


@ex.config
def config():

  return_result = False
  save_data_csv = None

  num_experiments = 10000
  num_threads = 1
  batch_size = 256
  save_interval = 50

  data = dict(
      npz_path=None,
      N=500,
      X_DIM=3,
      K=3,
      group_labels_generator_kind=
      "same",  # "same", "arithemetic_sequence", "custom(...)"
      group_X="same",  # "same", "group", "correlated", "custom(...)"
      T_star_factors=None,  # "None", "fixed(...)", "gamma(...)"
      exp_scale=3.5,
  )

  if data["npz_path"] is not None:
    npz_data = onp.load(data["npz_path"])
    _, X, _, group_labels = (npz_data["T"], npz_data["X"], npz_data["delta"],
                             npz_data["group_labels"])
    data["N"], data["X_DIM"] = X.shape
    data["K"] = onp.max(group_labels) + 1
    del X, group_labels, npz_data, _

  solver = dict(max_num_steps=40, loglik_eps=1e-5, score_norm_eps=1e-3)

  seed = 0
  key = jrandom.PRNGKey(seed)
  experiment_rand_key, data_generation_key = jrandom.split(key, 2)
  del key

  method = "unstratified_pooled"

  # groupped_configs
  distributed = dict(
      pt2_use_average_guess=data["npz_path"] is not None,
      hessian_use_taylor=True,
      taylor_order=1,
  )

  # meta_analysis
  meta_analysis = dict(univariate=False, use_only_dims=None)


@ex.main
def cov_experiment_main(data_generation_key, experiment_rand_key,
                        num_experiments, num_threads, batch_size, save_interval,
                        return_result):
  # pylint: disable=missing-function-docstring
  try:
    with tempfile.NamedTemporaryFile(mode="wb+") as result_file:
      res = cov_experiment(data_generation_key,
                           experiment_rand_key,
                           num_experiments=num_experiments,
                           num_threads=num_threads,
                           batch_size=batch_size,
                           save_interval=save_interval,
                           result_file=result_file)
      result_file.flush()
      ex.add_artifact(result_file.name, name="result")

    if return_result:
      print(res)
      return res
    return None
  except EndExperimentException:
    return None


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  run = ex.run_commandline()

  # import cProfile, pstats, io
  # from pstats import SortKey
  # pr = cProfile.Profile()
  # pr.enable()
  # pr.disable()
  # s = io.StringIO()
  # sortby = SortKey.CUMULATIVE
  # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
  # ps.print_stats()
  # print(s.getvalue())
