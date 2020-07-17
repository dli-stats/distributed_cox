"""Basic Common configs for experiments."""

from typing import Dict, Optional

import functools
import tempfile
import dataclasses
import importlib

import numpy as onp

from jax import jit, vmap
import jax.random as jrandom
import jax.tree_util as tu
import jax.numpy as np

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import varderiv.data as data
import varderiv.equations.eq1 as eq1

from varderiv.generic.model_solve import (solve_single, solve_distributed,
                                          cov_H, cov_robust,
                                          cov_group_correction)
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
#     X1 = jrandom.bernoulli(subkeys[1], p=0.3, shape=(N,)).astype(data.floatt)
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
  data_generation_key: onp.ndarray
  pt1: Optional[NewtonSolverResult]
  pt2: NewtonSolverResult
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


@ex.capture
def init_data_gen_fn(N, K, X_DIM, T_star_factors, group_labels_generator_kind,
                     group_X_same, exp_scale):
  """Initializes data generation."""

  if group_labels_generator_kind == "arithmetic_sequence":
    group_labels_generator_kind_kwargs = {"start_val": N * 2 // (K * (K + 1))}
  else:
    group_labels_generator_kind_kwargs = {}

  group_sizes = data.group_sizes_generator(
      N,
      K,
      group_labels_generator_kind=group_labels_generator_kind,
      **group_labels_generator_kind_kwargs)

  if not group_X_same:
    assert K == 3, "other than 3 groups not supported"
    X_generator = data.grouping_X_generator
  else:
    X_generator = data.default_X_generator

  if isinstance(T_star_factors, str) and T_star_factors.startswith("gamma"):
    # assume format gamma(*, *)
    gamma_args = T_star_factors[len("gamma"):]
    if len(gamma_args) == 0:
      gamma_args = (1., 1.)
    else:
      args = map(float, gamma_args.strip().split(","))
    T_star_factors = data.T_star_factors_gamma_gen(*args)
  elif T_star_factors == "fixed":
    T_star_factors = tuple((k + 1) / 2 for k in range(K))
  else:
    T_star_factors = None

  if exp_scale == 'inf':
    exp_scale = onp.inf

  return group_sizes, data.data_generator(N,
                                          X_DIM,
                                          group_sizes,
                                          exp_scale=exp_scale,
                                          T_star_factors=T_star_factors,
                                          X_generator=X_generator)


@ex.capture
def cov_experiment_init(eq, K, pt2_use_average_guess, solver_max_steps,
                        solver_eps, **experiment_params):
  num_single_args = 3

  is_groupped = eq != "eq1"
  eq_module = importlib.import_module("varderiv.equations.{}".format(eq))
  batch_log_likelihood_fn = getattr(eq_module,
                                    "batch_{}_log_likelihood".format(eq))
  log_likelihood_fn = getattr(eq_module, "{}_log_likelihood".format(eq))

  if not is_groupped:
    solve_fn = solve_single(log_likelihood_fn,
                            max_num_steps=solver_max_steps,
                            eps=solver_eps)
  else:
    single_log_likelihood_fn = eq1.eq1_log_likelihood
    solve_fn = solve_distributed(single_log_likelihood_fn,
                                 log_likelihood_fn,
                                 num_single_args,
                                 K=K,
                                 pt2_use_average_guess=pt2_use_average_guess,
                                 max_num_steps=solver_max_steps,
                                 eps=solver_eps)

  cov_fns = {}
  cov_fns["cov_H"] = cov_H()
  cov_fns["cov_robust"] = cov_robust(batch_log_likelihood_fn,
                                     num_single_args=num_single_args)

  if is_groupped:
    batch_single_log_likelihood_fn = eq1.batch_eq1_log_likelihood
    cov_fns["cov_group_correction"] = cov_group_correction(
        batch_single_log_likelihood_fn=batch_single_log_likelihood_fn,
        batch_distributed_log_likelihood_fn=batch_log_likelihood_fn,
        num_single_args=num_single_args,
        robust=False)
    cov_fns["cov_group_correction_robust"] = cov_group_correction(
        batch_single_log_likelihood_fn=batch_single_log_likelihood_fn,
        batch_distributed_log_likelihood_fn=batch_log_likelihood_fn,
        num_single_args=num_single_args,
        robust=True)

  group_sizes, gen = init_data_gen_fn()  # pylint: disable=no-value-for-parameter
  group_size = max(group_sizes)

  def solve_and_cov(data_generation_key):
    X, delta, beta, group_labels = gen(data_generation_key)
    if is_groupped:
      X_groups, delta_groups = data.group_data_by_labels(group_labels,
                                                         X,
                                                         delta,
                                                         K=K,
                                                         group_size=group_size)
      initial_beta_hat = beta
      initial_beta_k_hat = np.broadcast_to(beta, (K,) + beta.shape)
      model_args = (X, delta, initial_beta_hat, group_labels, X_groups,
                    delta_groups, initial_beta_k_hat)
    else:
      model_args = (X, delta, beta)

    sol = solve_fn(*model_args)

    if is_groupped:
      pt1_sol, pt2_sol = sol
      model_args = (X, delta, pt2_sol.guess, group_labels, X_groups,
                    delta_groups, pt1_sol.guess)
    else:
      pt1_sol, pt2_sol = None, sol
      model_args = (X, delta, sol.guess)

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
                                   check_fail_fn=lambda r: not r.sol.converged)


@ex.config
def config():
  num_experiments = 10000
  num_threads = 1
  batch_size = 256
  save_interval = 50

  N = 500
  X_DIM = 3
  K = 3
  group_labels_generator_kind = "same"
  group_X_same = True
  T_star_factors = None
  exp_scale = 3.5

  solver_max_steps = 80
  solver_eps = 1e-5

  seed = 0
  key = jrandom.PRNGKey(seed)
  experiment_rand_key, data_generation_key = jrandom.split(key, 2)
  del key

  eq = "eq1"

  # groupped_configs
  pt2_use_average_guess = False

  # meta_analysis
  univariate = False


@ex.main
def cov_experiment_main(data_generation_key, experiment_rand_key,
                        num_experiments, num_threads, batch_size,
                        save_interval):
  # pylint: disable=missing-function-docstring
  result_file = tempfile.NamedTemporaryFile(mode="wb+")
  ex.add_artifact(result_file.name, name="result")
  cov_experiment(data_generation_key,
                 experiment_rand_key,
                 num_experiments=num_experiments,
                 num_threads=num_threads,
                 batch_size=batch_size,
                 save_interval=save_interval,
                 result_file=result_file)


ex.captured_out_filter = apply_backspaces_and_linefeeds

if __name__ == '__main__':
  ex.run_commandline()
