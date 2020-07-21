"""Generic distributed solvers."""

from typing import Sequence, Union, Callable, Optional

import collections
import functools

import jax
from jax import jacfwd, vmap, hessian
import jax.numpy as np
import jax.scipy as scipy

from varderiv.generic.solver import solve_newton, NewtonSolverResult

DistributedModelSolverResult = collections.namedtuple(
    "DistributedModelSolverResult", "pt1 pt2")

MetaAnalysisResult = collections.namedtuple("MetaAnalysisResult",
                                            "guess converged")


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
                 **kwargs) -> NewtonSolverResult:
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


def _split_args(args, num_single_args):
  singe_static_args = args[:num_single_args]
  group_labels = args[num_single_args]
  distributed_args = args[num_single_args + 1:]
  return singe_static_args, group_labels, distributed_args


def solve_distributed(single_log_likelihood_or_score_fn,
                      distributed_log_likelihood_or_score_fn,
                      single_use_likelihood=True,
                      distributed_use_likelihood=True,
                      num_single_args: int = 1,
                      pt2_use_average_guess: bool = False,
                      K=1,
                      **kwargs) -> DistributedModelSolverResult:
  """Solves for distributed site."""

  assert num_single_args >= 0

  def solve_fun(*args):
    singe_static_args, group_labels, distributed_args = _split_args(
        args, num_single_args)
    pt1_sol = vmap(
        solve_single(single_log_likelihood_or_score_fn,
                     use_likelihood=single_use_likelihood,
                     **kwargs))(*distributed_args)

    pt1_guesses = pt1_sol.guess
    pt1_converged = np.all(pt1_sol.converged, axis=0)

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
    pt2_sol = pt2_sol._replace(converged=pt1_converged & pt2_sol.converged)
    return DistributedModelSolverResult(pt1=pt1_sol, pt2=pt2_sol)

  return solve_fun


def solve_meta_analysis(single_log_likelihood_or_score_fn,
                        use_likelihood: bool = True,
                        use_only_dims: Optional[Sequence] = None,
                        univariate: bool = False,
                        **kwargs) -> DistributedModelSolverResult:

  def solve_fun(*args):
    sol = vmap(
        solve_single(single_log_likelihood_or_score_fn,
                     use_likelihood=use_likelihood,
                     **kwargs))(*args)
    converged = np.all(sol.converged, axis=0)
    I_diag_wo_last = -sol.hessian

    if use_only_dims is not None:
      I_diag_wo_last = np.take(np.take(I_diag_wo_last, use_only_dims, axis=1),
                               use_only_dims,
                               axis=2)
      pt1_guesses = np.take(sol.guess, use_only_dims, axis=-1)
    else:
      pt1_guesses = sol.guess

    if univariate:
      wk = 1. / np.diagonal(np.linalg.inv(I_diag_wo_last), axis1=-2, axis2=-1)
      guess = np.einsum("kp,kp->p", wk, pt1_guesses,
                        optimize="optimal") / np.sum(wk, axis=0)
    else:
      guess = np.linalg.solve(
          np.sum(I_diag_wo_last, axis=0),
          np.einsum("Kab,Kb->a",
                    I_diag_wo_last,
                    pt1_guesses,
                    optimize='optimal'))
    return DistributedModelSolverResult(pt1=sol,
                                        pt2=MetaAnalysisResult(
                                            guess=guess, converged=converged))

  return solve_fun


def cov_meta_analysis(use_only_dims: Optional[Sequence] = None,
                      univariate=False):

  def wrapped(sol: DistributedModelSolverResult):
    I_diag_wo_last = -sol.pt1.hessian
    if use_only_dims is not None:
      I_diag_wo_last = np.take(np.take(I_diag_wo_last, use_only_dims, axis=1),
                               use_only_dims,
                               axis=2)
    if univariate:
      wk = 1. / np.diagonal(np.linalg.inv(I_diag_wo_last), axis1=-2, axis2=-1)
      X_DIM = sol.pt2.guess.shape[0]
      cov = np.zeros((X_DIM, X_DIM), dtype=sol.pt2.guess.dtype)
      cov = jax.ops.index_update(cov, np.diag_indices(X_DIM),
                                 1. / np.sum(wk, axis=0))
    else:
      cov = np.linalg.inv(np.sum(I_diag_wo_last, axis=0))
    return cov

  return wrapped


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
    batch_score_fn = jacfwd(batch_log_likelihood_or_score_fn,
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
    return A_inv @ B @ A_inv.T

  return wrapped


def cov_group_correction(
    batch_single_log_likelihood_or_score_fn: Callable,
    batch_distributed_log_likelihood_or_score_fn: Callable,
    distributed_cross_hessian_fn: Optional[Callable] = None,
    num_single_args: int = 1,
    batch_single_use_likelihood=True,
    batch_distributed_use_likelihood=True,
    robust=False):
  """Computes covariance with grouped correction."""

  if batch_single_use_likelihood:
    batch_single_score_fn = jacfwd(batch_single_log_likelihood_or_score_fn,
                                   num_single_args - 1)
  else:
    batch_single_score_fn = batch_single_log_likelihood_or_score_fn

  if batch_distributed_use_likelihood:
    batch_distributed_score_fn = jacfwd(
        batch_distributed_log_likelihood_or_score_fn, num_single_args - 1)
    if distributed_cross_hessian_fn is None:
      distributed_cross_hessian_fn = hessian(
          sum_log_likelihood(batch_distributed_log_likelihood_or_score_fn), -1)
  else:
    batch_distributed_score_fn = batch_distributed_log_likelihood_or_score_fn
    if distributed_cross_hessian_fn is None:
      distributed_cross_hessian_fn = jacfwd(
          sum_score(batch_distributed_score_fn), -1)

  def wrapped(sol: DistributedModelSolverResult, *args):
    pt1_sol, pt2_sol = sol
    distributed_args = args[num_single_args + 1:-1] + (pt1_sol.guess,)

    I_diag_wo_last = -pt1_sol.hessian
    I_diag_last = -pt2_sol.hessian

    I_row_wo_last = -distributed_cross_hessian_fn(*args)

    I_diag_inv_wo_last = np.linalg.inv(I_diag_wo_last)
    I_diag_inv_last = np.linalg.inv(I_diag_last)

    if robust:
      pt1_batch_scores = vmap(batch_single_score_fn)(*distributed_args)
      pt2_batch_scores = batch_distributed_score_fn(*args)  # batch_dims x X_dim

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
      sb2s = sb1s.T  # pylint: disable=no-member
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
