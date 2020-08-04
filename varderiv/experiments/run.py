"""Basic Common configs for experiments."""

from typing import Dict, Optional, Union

import collections
import functools
import tempfile
import dataclasses
import importlib
import itertools

from frozendict import frozendict

import numpy as onp

from jax import jit, vmap, jacrev, jacfwd
import jax.random as jrandom
import jax.tree_util as tu
import jax.numpy as np

from simpleeval import EvalWithCompoundTypes

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import varderiv.data as vdata
import varderiv.equations.eq1 as eq1

import varderiv.generic.modeling as modeling
from varderiv.generic.solver import NewtonSolverResult

import varderiv.experiments.utils as utils

# pylint: disable=unused-variable
# pylint: disable=missing-function-docstring

# def grouping_Xi_generator_2(N, dim, key, group_label=0):
#   if group_label == 0:
#     bernoulli_theta, normal_mean, normal_variance = 0.5, 0, 1
#   elif group_label == 1:
#     bernoulli_theta, normal_mean, normal_variance = 0.3, 1, 0.5
#   elif group_label == 2:
#     bernoulli_theta, normal_mean, normal_variance = 0.7, -1, 1.5
#   return jax.lax.cond(
#       dim % 2 == 0,
#       key, \
#       lambda key: jrandom.bernoulli(key, p=bernoulli_theta,
#                                     shape=(N,)).astype(np.float32),
#       key, \
#       lambda key: jrandom.normal(key, shape=(N,))) * normal_variance + normal_mean

# grouping_X_generator_2 = functools.partial(X_group_generator_indep_dim,
#                                            Xi_generator=grouping_Xi_generator_2)

# def X_group_generator_joint(N, X_dim, key, group_label=0):
#   assert X_dim == 3
#   if group_label == 0:
#     return default_X_generator(N, X_dim, key, group_label=0)
#   elif group_label == 1:
#     key, *subkeys = jrandom.split(key, 3)
#     X02 = jrandom.multivariate_normal(subkeys[0],
#                                       np.array([0, 0]),
#                                       np.array([[1, 0.25], [0.25, 0.25]]),
#                                       shape=(N,))
#     X1 = jrandom.bernoulli(subkeys[1], p=0.3, shape=(N,)).astype(vdata.floatt)
#     return np.stack(([X02[:, 0], X1, X02[:, 1]]), axis=1)
#   else:
#     key, *subkeys = jrandom.split(key, 3)
#     X02 = jrandom.multivariate_normal(subkeys[0],
#                                       np.array([0, 0]),
#                                       np.array([[1.5, 0.5], [0.5, 0.25]]),
#                                       shape=(N,))
#     X1 = jrandom.normal(subkeys[1], shape=(N,)) * 1.5
#     return np.stack(([X02[:, 0], X1, X02[:, 1]]), axis=1)

ex = Experiment("varderiv")


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


def compute_results_averaged(result: ExperimentResult):
  """
  Args:
    - result: the experiment result object.
    - A function given a result.sol, returns the beta
    - A dict from str to function. The string represents a key
      to plot the analytical cov. The function returns the cov
      matrices from result.cov.
  """
  all_covs = collections.OrderedDict()

  converged_idxs = result.sol.converged
  if result.is_groupped:
    pt1_converged_idxs = onp.all(result.pt1.converged, axis=1)
    converged_idxs &= pt1_converged_idxs

  beta = result.sol.guess

  beta_empirical_nan_idxs = onp.any(onp.isnan(beta), axis=1)

  keep_idxs = converged_idxs & ~beta_empirical_nan_idxs
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
  all_covs["empirical"] = cov_empirical

  for cov_name, cov_analyticals in result.covs.items():
    cov_analyticals = cov_analyticals[keep_idxs]
    cov_analytical = onp.mean(cov_analyticals, axis=0)
    all_covs[cov_name] = cov_analytical

  return beta_hat, all_covs


def eval_get_group_X_generator(group_X: str):
  evaluator = EvalWithCompoundTypes(
      functions={
          'normal':
              vdata.normal,
          'bernoulli':
              vdata.bernoulli,
          'custom':
              lambda gdists, dims, weights: functools.partial(
                  vdata.make_X_generator,
                  g_dists=gdists,
                  correlated_dims=dims,
                  correlated_weights=weights),
      },
      names={
          'group': vdata.grouping_X_generator_K3,
          'same': vdata.default_X_generator,
          'correlated': vdata.correlated_X_generator
      })
  return evaluator.eval(group_X)


@ex.capture(prefix="data")
@functools.lru_cache(maxsize=None)
def init_data_gen_fn(N, K, X_DIM, T_star_factors, group_labels_generator_kind,
                     group_X, exp_scale):
  """Initializes data generation."""

  if group_labels_generator_kind == "arithmetic_sequence":
    group_labels_generator_kind_kwargs = {"start_val": N * 2 // (K * (K + 1))}
  else:
    group_labels_generator_kind_kwargs = {}

  group_sizes = vdata.group_sizes_generator(
      N,
      K,
      group_labels_generator_kind=group_labels_generator_kind,
      **group_labels_generator_kind_kwargs)

  X_generator = eval_get_group_X_generator(group_X)

  evaluator = EvalWithCompoundTypes()

  def parse_float_tuple(s, prefix, default):
    s = s[len(prefix):].strip()
    if len(s) == 0:
      return default
    assert s[0] == "(" and s[-1] == ")"
    return tuple(map(float, evaluator.eval(s)))

  if isinstance(T_star_factors, str) and T_star_factors.startswith("gamma"):
    gamma_args = parse_float_tuple(T_star_factors, "gamma", (1., 1.))
    T_star_factors = vdata.T_star_factors_gamma_gen(*gamma_args)
  elif T_star_factors == "fixed":
    T_star_factors = parse_float_tuple(T_star_factors, "fixed",
                                       tuple((k + 1) / 2 for k in range(K)))
  else:
    T_star_factors = None

  if exp_scale == 'inf':
    exp_scale = onp.inf

  return group_sizes, vdata.data_generator(N,
                                           X_DIM,
                                           group_sizes,
                                           exp_scale=exp_scale,
                                           T_star_factors=T_star_factors,
                                           X_generator=X_generator)


def freezeargs(func):
  """Transform mutable dictionnary
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
def cov_experiment_init(eq, data, distributed, solver, meta_analysis,
                        **experiment_params):
  del experiment_params
  K = data["K"]
  num_single_args = 3

  if eq == "meta_analysis":
    solve_fn = functools.partial(modeling.solve_meta_analysis,
                                 eq1.eq1_log_likelihood,
                                 use_likelihood=True,
                                 **meta_analysis)
  else:
    eq_mod = importlib.import_module("varderiv.equations.{}".format(eq))
    if eq in ("eq1", "eq3"):
      batch_log_likelihood_or_score_fn = getattr(
          eq_mod, "batch_{}_log_likelihood".format(eq))
      log_likelihood_fn = getattr(eq_mod, "{}_log_likelihood".format(eq))
      use_likelihood = True
      solve_fn = functools.partial(modeling.solve_single,
                                   log_likelihood_fn,
                                   use_likelihood=use_likelihood)
    elif eq in ("eq2", "eq4"):
      batch_log_likelihood_or_score_fn = getattr(eq_mod,
                                                 "batch_{}_score".format(eq))
      log_likelihood_or_score_fn = getattr(eq_mod, "{}_score".format(eq))
      use_likelihood = False
      if distributed["hessian_use_taylor2"]:
        hessian_fn = getattr(eq_mod, "hessian_taylor2")
      else:
        hessian_fn = jacfwd(log_likelihood_or_score_fn, num_single_args - 1)
      solve_fn = functools.partial(
          modeling.solve_distributed,
          eq1.eq1_log_likelihood,
          distributed_hessian_fn=hessian_fn,
          num_single_args=num_single_args,
          K=K,
          pt2_use_average_guess=distributed["pt2_use_average_guess"],
          single_use_likelihood=True)
      solve_fn = functools.partial(solve_fn,
                                   log_likelihood_or_score_fn,
                                   distributed_use_likelihood=use_likelihood)

  solve_fn = solve_fn(**solver)

  cov_fns = {}

  if eq == "meta_analysis":
    cov_fns["cov:meta_analysis"] = modeling.cov_meta_analysis(**meta_analysis)
  else:
    for (group_correction, sandwich_robust, sandwich_robust_sum_group_first,
         cox_correction) in itertools.product(*[(True, False)] * 4):
      # Some non-sensical situations
      if not sandwich_robust and cox_correction:
        continue
      if not sandwich_robust and sandwich_robust_sum_group_first:
        continue
      if group_correction and eq in ("eq1", "eq3"):
        continue
      if sandwich_robust_sum_group_first and eq in ("eq1", "eq2"):
        continue
      if sandwich_robust_sum_group_first:  # disabled because it's too bad
        continue
      batch_robust_cox_correction_score = getattr(
          eq_mod, "batch_{}_robust_cox_correction_score".format(eq))
      if group_correction:
        batch_score = getattr(eq_mod, "batch_{}_score".format(eq))
        cov_fn = modeling.cov_group_correction(
            (eq1.batch_eq1_robust_cox_correction_score \
            if cox_correction else eq1.batch_eq1_log_likelihood),
            (batch_robust_cox_correction_score \
            if cox_correction else batch_score),
            distributed_cross_hessian_fn=jacrev(
                getattr(eq_mod, "{}_score".format(eq)), -1),
            batch_single_use_likelihood=not cox_correction,
            batch_distributed_use_likelihood=False,
            num_single_args=num_single_args,
            robust=sandwich_robust,
            robust_sum_group_first=sandwich_robust_sum_group_first)
      elif sandwich_robust:
        cov_batch_log_likelihood_or_score_fn = (
            batch_robust_cox_correction_score
            if cox_correction else batch_log_likelihood_or_score_fn)
        cov_fn = modeling.cov_robust(
            batch_log_likelihood_or_score_fn=
            cov_batch_log_likelihood_or_score_fn,
            use_likelihood=(use_likelihood and not cox_correction),
            num_single_args=num_single_args,
            sum_group_first=sandwich_robust_sum_group_first)
      else:
        cov_fn = modeling.cov_H()
      cov_name = ("cov:{}group_correction|{}sandwich"
                  "|{}cox_correction|{}sum_first").format(*[
                      "" if v else "no_"
                      for v in (group_correction, sandwich_robust,
                                cox_correction, sandwich_robust_sum_group_first)
                  ])
      cov_fns[cov_name] = cov_fn

  group_sizes, gen = init_data_gen_fn()  # pylint: disable=no-value-for-parameter
  group_size = max(group_sizes)

  def solve_and_cov(data_generation_key):
    X, delta, beta, group_labels = gen(data_generation_key)
    initial_beta_hat = beta
    if eq != "eq1":
      X_groups, delta_groups = vdata.group_data_by_labels(group_labels,
                                                          X,
                                                          delta,
                                                          K=K,
                                                          group_size=group_size)

      initial_beta_k_hat = np.broadcast_to(beta, (K,) + beta.shape)

    if eq == "eq1":
      pt1_sol = None
      pt2_sol = sol = solve_fn(X, delta, initial_beta_hat)
      model_args = (X, delta, sol.guess)
    elif eq == "eq3":
      pt1_sol = None
      pt2_sol = sol = solve_fn(X_groups, delta_groups, initial_beta_hat)
      model_args = (X_groups, delta_groups, sol.guess)
    elif eq in ("eq2", "eq4"):
      pt1_sol, pt2_sol = sol = solve_fn(X, delta, initial_beta_hat,
                                        group_labels, X_groups, delta_groups,
                                        initial_beta_k_hat)
      model_args = (X, delta, pt2_sol.guess, group_labels, X_groups,
                    delta_groups, pt1_sol.guess)
    elif eq == "meta_analysis":
      sol = solve_fn(X_groups, delta_groups, initial_beta_k_hat)
      pt1_sol, pt2_sol = sol.pt1, sol.pt2
      model_args = tuple()

    cov_results = {}
    for cov_name, cov_fn in cov_fns.items():
      cov_results[cov_name] = cov_fn(sol, *model_args)

    return ExperimentResult(data_generation_key=data_generation_key,
                            pt1=pt1_sol,
                            pt2=pt2_sol,
                            covs=cov_results)

  return {"solve_and_cov": jit(vmap(solve_and_cov))}


@ex.capture
def cov_experiment_core(rnd_keys, solve_and_cov=None):
  assert solve_and_cov is not None
  key, data_generation_key = map(np.array, zip(*rnd_keys))
  experiment_result = solve_and_cov(data_generation_key)
  return experiment_result


cov_experiment = functools.partial(utils.run_cov_experiment,
                                   cov_experiment_init,
                                   cov_experiment_core,
                                   check_ok_fn=lambda r: r.sol.converged)


@ex.config
def config():
  return_result = False

  num_experiments = 10000
  num_threads = 1
  batch_size = 256
  save_interval = 50

  data = dict(
      N=500,
      X_DIM=3,
      K=3,
      group_labels_generator_kind="same",  # "same", "arithemetic_sequence"
      group_X="same",  # "same", "group", "correlated", "custom(...)"
      T_star_factors=None,  # "None", "fixed(...)", "gamma(...)"
      exp_scale=3.5,
  )
  solver = dict(max_num_steps=40, loglik_eps=1e-5, score_norm_eps=1e-3)

  seed = 0
  key = jrandom.PRNGKey(seed)
  experiment_rand_key, data_generation_key = jrandom.split(key, 2)
  del key

  eq = "eq1"

  # groupped_configs
  distributed = dict(pt2_use_average_guess=False, hessian_use_taylor2=True)

  # meta_analysis
  meta_analysis = dict(univariate=False, use_only_dims=None)


@ex.main
def cov_experiment_main(data_generation_key, experiment_rand_key,
                        num_experiments, num_threads, batch_size, save_interval,
                        return_result):
  # pylint: disable=missing-function-docstring
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
    return res
  else:
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
