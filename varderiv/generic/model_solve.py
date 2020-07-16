"""Generic distributed solvers."""

from typing import Sequence

import collections
import functools

from jax import jacrev
import jax.numpy as np
import jax.scipy as scipy

from varderiv.generic.solver import solve_newton, NewtonSolverResult

DistributedModelSolverResult = collections.namedtuple(
    "DistributedModelSolverResult", "pt1 pt2")


def solve_single(single_log_likelihood_fn, **kwargs):
  """Solves for single site."""

  def solve_fun(*args):
    singe_static_args = args[:-1]
    initial_guess = args[-1]
    sol = solve_newton(
        functools.partial(single_log_likelihood_fn, *singe_static_args),
        initial_guess, **kwargs)
    return sol

  return solve_fun


def solve_distributed(single_log_likelihood_fn,
                      distributed_log_likelihood_fn,
                      num_single_args: int = 1,
                      pt2_use_average_guess: bool = False,
                      **kwargs):
  """Solves for distributed site."""

  assert num_single_args >= 0

  def solve_fun(*args):
    num_param_args = 1  # only support one single parameter argument for now
    singe_static_args = args[:num_single_args - num_param_args]
    pt1_initial_guess = args[num_param_args:num_single_args]
    pt1_sol = solve_newton(
        functools.partial(single_log_likelihood_fn, *singe_static_args),
        pt1_initial_guess, **kwargs)

    group_labels = args[num_single_args]
    distributed_args = args[num_single_args + 1:]

    pt1_guesses = pt1_sol.guess

    def pt2_loglik(initial_guess):
      return distributed_log_likelihood_fn(*singe_static_args, initial_guess,
                                           group_labels, *distributed_args,
                                           pt1_guesses)

    if pt2_use_average_guess:
      pt2_initial_guess = np.mean(pt1_guesses, axis=0)
    else:
      pt2_initial_guess = pt1_initial_guess

    pt2_sol = solve_newton(pt2_loglik, pt2_initial_guess, *kwargs)

    return DistributedModelSolverResult(pt1=pt1_sol, pt2=pt2_sol)

  return solve_fun


def cov_H(sol: NewtonSolverResult):

  def wrapped(*args):
    del args
    return scipy.linalg.inv(-sol.hessian)

  return wrapped


def cov_robust(batch_log_likelihood_fn, sol: NewtonSolverResult):

  def wrapped(*args):
    A_inv = scipy.linalg.inv(-sol.hessian)
    batch_score = jacrev(batch_log_likelihood_fn)(*args)
    B = np.einsum("ni,nj->ij", batch_score, batch_score, optimize="optimal")
    return A_inv @ B @ A_inv

  return wrapped


def cov_group_correction(batch_single_log_likelihood_fn,
                         batch_distributed_log_likelihood_fn,
                         num_single_args: int = 1,
                         pt1_sol: Sequence[NewtonSolverResult],
                         pt2_sol: NewtonSolverResult):

  def wrapped(*args):
    distributed_args = args[num_single_args + 1:]
    pt1_batch_scores = vmap(batch_single_log_likelihood_fn)(*distributed_args)
    pt2_batch_scores = batch_distributed_log_likelihood_fn(*args)
    B_diag_wo_last = np.einsum("kbi,kbj->kij",
                               pt1_batch_scores,
                               pt1_batch_scores,
                               optimize="optimal")
    B_diag_last = np.einsum("ki,kj->ij",
                            pt2_batch_scores,
                            pt2_batch_scores,
                            optimize="optimal")
    B_row_wo_last = np.einsum("kbi,kbj->kij",
                              pt2_W_grouped,
                              pt1_W,
                              optimize="optimal")
