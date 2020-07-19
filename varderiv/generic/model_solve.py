"""Generic distributed solvers."""

from typing import Sequence, Union, Callable, Optional

import collections
import functools

from jax import jacrev, jacfwd, vmap, hessian
import jax.numpy as np
import jax.scipy as scipy

from varderiv.generic.solver import solve_newton, NewtonSolverResult

DistributedModelSolverResult = collections.namedtuple(
    "DistributedModelSolverResult", "pt1 pt2")


def sum_fn(fun, ndims=0):
  """Helper for summing fun. """

  def wrapped(*args):
    batch_loglik = fun(*args)
    return np.sum(
        batch_loglik.reshape((-1,) +
                             batch_loglik.shape[-ndims +
                                                len(batch_loglik.shape):]),
        axis=0)

  return wrapped


sum_log_likelihood = functools.partial(sum_fn, ndims=0)
sum_score = functools.partial(sum_fn, ndims=1)


def solve_single(single_log_likelihood_or_score_fn,
                 use_likelihood=True,
                 **kwargs):
  """Solves for single site."""

  def solve_fun(*args):
    singe_static_args = args[:-1]
    initial_guess = args[-1]
    log_likelihood_or_score_fn = functools.partial(
        single_log_likelihood_or_score_fn,
        *singe_static_args,
    )
    sol = solve_newton(log_likelihood_or_score_fn,
                       initial_guess,
                       use_likelihood=use_likelihood,
                       **kwargs)
    return sol

  return solve_fun


def solve_distributed(single_log_likelihood_or_score_fn,
                      distributed_log_likelihood_or_score_fn,
                      single_use_likelihood=True,
                      distributed_use_likelihood=True,
                      num_single_args: int = 1,
                      pt2_use_average_guess: bool = False,
                      K=1,
                      **kwargs):
  """Solves for distributed site."""

  assert num_single_args >= 0

  def solve_fun(*args):
    # only support one single parameter argument for now
    singe_static_args = args[:num_single_args]
    group_labels = args[num_single_args]
    distributed_args = args[num_single_args + 1:]
    pt1_sol = vmap(
        solve_single(single_log_likelihood_or_score_fn,
                     use_likelihood=single_use_likelihood,
                     **kwargs))(*distributed_args)

    pt1_guesses = pt1_sol.guess

    def pt2_loglik_or_score(initial_guess):
      return distributed_log_likelihood_or_score_fn(*singe_static_args[:-1],
                                                    initial_guess, group_labels,
                                                    *distributed_args[:-1],
                                                    pt1_guesses)

    if pt2_use_average_guess:
      pt2_initial_guess = np.mean(pt1_guesses, axis=0)
    else:
      pt2_initial_guess = args[num_single_args - 1]

    pt2_sol = solve_newton(pt2_loglik_or_score,
                           pt2_initial_guess,
                           use_likelihood=distributed_use_likelihood,
                           **kwargs)

    return DistributedModelSolverResult(pt1=pt1_sol, pt2=pt2_sol)

  return solve_fun


def cov_H():

  def wrapped(sol: Union[NewtonSolverResult, DistributedModelSolverResult],
              *args):
    del args
    if isinstance(sol, DistributedModelSolverResult):
      sol = sol.pt2
    return scipy.linalg.inv(-sol.hessian)

  return wrapped


def cov_robust(batch_log_likelihood_or_score_fn,
               num_single_args: int = 1,
               use_likelihood=True):

  if use_likelihood:
    batch_score_fn = jacrev(batch_log_likelihood_or_score_fn,
                            argnums=num_single_args - 1)
  else:
    batch_score_fn = batch_log_likelihood_or_score_fn

  def wrapped(sol: Union[NewtonSolverResult, DistributedModelSolverResult],
              *args):
    if isinstance(sol, DistributedModelSolverResult):
      sol = sol.pt2
    A_inv = scipy.linalg.inv(-sol.hessian)
    batch_score = batch_score_fn(*args)
    batch_score = batch_score.reshape((-1, batch_score.shape[-1]))
    B = np.einsum("ni,nj->ij", batch_score, batch_score, optimize="optimal")
    return A_inv @ B @ A_inv

  return wrapped


def cov_group_correction(batch_single_log_likelihood_or_score_fn: Callable,
                         batch_distributed_log_likelihood_or_score_fn: Callable,
                         distributed_cross_hessian_fn: Optional[Callable],
                         num_single_args: int = 1,
                         batch_single_use_likelihood=True,
                         batch_distributed_use_likelihood=True,
                         robust=False):
  """Computes covariance with grouped correction."""

  if batch_single_use_likelihood:
    batch_single_score_fn = jacrev(batch_single_log_likelihood_or_score_fn,
                                   num_single_args - 1)
  else:
    batch_single_score_fn = batch_single_log_likelihood_or_score_fn

  if batch_distributed_use_likelihood:
    batch_distributed_score_fn = jacrev(
        batch_distributed_log_likelihood_or_score_fn, num_single_args - 1)
  else:
    batch_distributed_score_fn = batch_distributed_log_likelihood_or_score_fn

  if distributed_cross_hessian_fn is None:
    distributed_cross_hessian_fn = jacrev(sum_score(batch_distributed_score_fn),
                                          -1)

  def wrapped(sol: DistributedModelSolverResult, *args):
    pt1_sol, pt2_sol = sol
    distributed_args = args[num_single_args + 1:-1] + (pt1_sol.guess,)

    I_diag_wo_last = pt1_sol.hessian
    I_diag_last = pt2_sol.hessian

    I_row_wo_last = -distributed_cross_hessian_fn(*args)

    I_diag_inv_wo_last = np.linalg.inv(-I_diag_wo_last)
    I_diag_inv_last = np.linalg.inv(-I_diag_last)

    if robust:
      pt1_batch_scores = vmap(batch_single_score_fn)(*distributed_args)
      pt2_batch_scores = batch_distributed_score_fn(*args)  # K x S x X_dim

      B_diag_wo_last = np.einsum("kbi,kbj->kij",
                                 pt1_batch_scores,
                                 pt1_batch_scores,
                                 optimize="optimal")
      B_diag_last = np.einsum("ksi,ksj->ij",
                              pt2_batch_scores,
                              pt2_batch_scores,
                              optimize="optimal")
      B_row_wo_last = np.einsum("kbi,kbj->kij",
                                pt2_batch_scores,
                                pt1_batch_scores,
                                optimize="optimal")

      S = np.einsum("ab,bBc,Bcd->Bad",
                    I_diag_inv_last,
                    I_row_wo_last,
                    I_diag_inv_wo_last,
                    optimize="optimal")

      sas = np.einsum("Bab,Bbc,Bdc->ad",
                      S,
                      B_diag_wo_last,
                      S,
                      optimize="optimal")
      sb1s = np.einsum("ab,Bbc,Bdc->ad",
                       I_diag_inv_last,
                       B_row_wo_last,
                       S,
                       optimize="optimal")
      sb2s = np.einsum("Bab,Bbc,dc->ad",
                       S,
                       B_row_wo_last,
                       I_diag_inv_last,
                       optimize="optimal")
      scs = np.einsum('ab,bc,dc->ad',
                      I_diag_inv_last,
                      B_diag_last,
                      I_diag_inv_last,
                      optimize='optimal')
      cov = sas - sb1s - sb2s + scs

    else:

      S = np.einsum("ab,bBc->Bac",
                    I_diag_inv_last,
                    I_row_wo_last,
                    optimize="optimal")

      cov = np.einsum(
          "Bab,Bbc,Bdc->ad", S, I_diag_inv_wo_last, S,
          optimize='optimal') + I_diag_inv_last

    return cov

  return wrapped
