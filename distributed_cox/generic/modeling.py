"""Generic distributed solvers."""

from typing import Sequence, Union, Callable, Optional

import collections
import functools

import jax
from jax import jacfwd, vmap, jacrev
import jax.numpy as jnp
import jax.scipy as scipy
import jax.scipy.optimize as scipy_optimize
from oryx.core import sow, reap

from distributed_cox.generic.solver import solve_newton, NewtonSolverResult

DistributedModelSolverResult = collections.namedtuple(
    "DistributedModelSolverResult", "pt1 pt2")

MetaAnalysisResult = collections.namedtuple("MetaAnalysisResult",
                                            "guess converged")


def sum_fn(fun, ndims=0):
  """Higher order helper for summing the result of fun."""

  @functools.wraps(fun)
  def wrapped(*args):
    batch_loglik = fun(*args)
    return jnp.sum(
        batch_loglik.reshape((-1,) +
                             batch_loglik.shape[-ndims +
                                                len(batch_loglik.shape):]),
        axis=0)

  return wrapped


#: sums an array of log likehiloods scalars
sum_loglik = functools.partial(sum_fn, ndims=0)
#: sums an array of score vectors
sum_score = functools.partial(sum_fn, ndims=1)
#: sums an array of hessian matrices
sum_hessian = functools.partial(sum_fn, ndims=2)


def model_temporaries(tag: str):
  """Utility for marking/collecting model's intermediates.

  Given a tag, it returns a pair of function ``mark`` and ``collect``. These are
  simple wrappers around :py:func:`oryx.core.sow` and :py:func:`oryx.core.reap`.

  Args:
    tag: a unique tag identifying a model.

  Returns:
    the functions ``mark`` and ``collect``.
  """

  def mark(e, name):
    return sow(e, tag=tag, name=name, mode="clobber")

  def collect(fun, names: Union[Sequence[str]]):
    """Collect marked intermediates.

    Args:
      names: Either a sequence of or a single string name(s). If a single name,
      return the function that returns the sole marked intermediate.
      Otherwise, return all the intermediates (in the order of the names).
    """
    if isinstance(names, str):
      names = [names]
      ret_single = True
    else:
      ret_single = False

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
      temps = reap(fun, tag=tag, allowlist=names)(*args, **kwargs)
      if ret_single:
        return temps[names[0]]
      return [temps[name] for name in names]

    return wrapped

  return mark, collect


def solve_single(single_log_likelihood_or_score_fn,
                 use_likelihood=True,
                 **kwargs):
  """Solves for single site.

  Solves the parameters that result in either 1) the maximum log likelihood of
  the model, or 2) the 0-point of the score function.

  Args:
    single_log_likelihood_or_score_fn: a log likelihood function or a score
      function of the model.
    use_likelihood: whether ``single_log_likelihood_or_score_fn`` is a log
      likelihood function.
    kwargs: arguments to be passed into the solver.

  Returns:
    a function that takes in the same parameters as
    ``single_log_likelihood_or_score_fn``, and returns the solved results.
  """

  def solve_fun(*args) -> NewtonSolverResult:
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


def solve_distributed(single_log_likelihood_or_score_fn: Callable,
                      distributed_log_likelihood_or_score_fn: Callable,
                      distributed_hessian_fn: Optional[Callable] = None,
                      single_use_likelihood=True,
                      distributed_use_likelihood=True,
                      num_single_args: int = 1,
                      pt2_use_average_guess: bool = False,
                      pt2_zero_use_pt1_average_guess: bool = False,
                      K=1,
                      pt2_solver: Optional[str] = "newton",
                      **kwargs) -> DistributedModelSolverResult:
  """Solves for distributed site.

  Args:
    single_log_likelihood_or_score_fn: a log likelihood function or a score
      function for the single site version of the model.
    distributed_log_likelihood_or_score_fn: a log likelihood function or a score
      function for the distributed version of the model.
    distributed_hessian_fn: an optional hessian function for the distributed
      version of the model. If not provided, use automatic differentiation to
      derive this function based on ``distributed_log_likelihood_or_score_fn``.
    single_use_likelihood: whether ``single_log_likelihood_or_score_fn`` is a
      log likelihood function.
    distributed_use_likelihood: whether
      ``distributed_log_likelihood_or_score_fn`` is a log likelihood function.
    num_single_args: number of arguments for
      ``single_log_likelihood_or_score_fn``.
    pt2_initial_use_average_guess: wether to use the average from solving the
      single sites, as the initial guess for solving the distributed model.
    pt2_zero_use_pt1_average_guess: wether to use the average from solving the
      single sites, as the taylor expansion point for solving the distributed
      model. Defaults to False, which will use individual sites solutions.
    K: the number of sites.
    pt2_solver: the solver to use for solving the distributed version. Defaults
      to the newton raphson solver.
    kwargs: arguments to be passed into the solver.

  Returns:
    a function that takes in the parameters as
    ``distributed_log_likelihood_or_score_fn``, and returns the solved results.
  """

  assert num_single_args >= 0

  def solve_fun(*args):
    single_static_args, group_labels, distributed_args = _split_args(
        args, num_single_args)
    pt1_sol = vmap(
        solve_single(single_log_likelihood_or_score_fn,
                     use_likelihood=single_use_likelihood,
                     **kwargs))(*distributed_args)

    pt1_converged = jnp.all(pt1_sol.converged, axis=0)

    def wrap_fun_single_arg(fun):

      def wrapped(initial_guess):
        if pt2_zero_use_pt1_average_guess:
          ss = jnp.mean(pt1_sol.guess, axis=0)
          pt1_guesses = jnp.broadcast_to(ss, (K,) + ss.shape)
        else:
          pt1_guesses = pt1_sol.guess
        return fun(*single_static_args[:-1], initial_guess, group_labels,
                   *distributed_args[:-1], pt1_guesses)

      return wrapped

    if pt2_use_average_guess:
      pt2_initial_guess = jnp.mean(pt1_sol.guess, axis=0)
    else:
      pt2_initial_guess = args[num_single_args - 1]

    if pt2_solver == "newton":
      pt2_sol = solve_newton(
          wrap_fun_single_arg(distributed_log_likelihood_or_score_fn),
          pt2_initial_guess,
          hessian_fn=wrap_fun_single_arg(distributed_hessian_fn)
          if distributed_hessian_fn is not None else None,
          use_likelihood=distributed_use_likelihood,
          **kwargs)
    elif pt2_solver == "bfgs":
      bfgs_sol = scipy_optimize.minimize(
          wrap_fun_single_arg(distributed_log_likelihood_or_score_fn),
          pt2_initial_guess,
          method="bfgs",
          options=dict(
              maxiter=kwargs.get("max_num_steps", 40),
              gtol=kwargs.get("loglik_eps", 1e-5),
              line_search_maxiter=10,
          ),
      )
      converged = bfgs_sol.fun < kwargs.get("loglik_eps", 1e-5)
      pt2_sol = NewtonSolverResult(guess=bfgs_sol.x,
                                   loglik=bfgs_sol.fun,
                                   score=bfgs_sol.jac,
                                   hessian=bfgs_sol.hess_inv,
                                   step=bfgs_sol.nit,
                                   converged=converged)
    else:
      raise ValueError("Solver must be one of {}".format(["bfgs", "newton"]))
    pt2_sol = pt2_sol._replace(converged=pt1_converged & pt2_sol.converged)
    return DistributedModelSolverResult(pt1=pt1_sol, pt2=pt2_sol)

  return solve_fun


def solve_meta_analysis(single_log_likelihood_or_score_fn,
                        use_likelihood: bool = True,
                        use_only_dims: Optional[Sequence[int]] = None,
                        univariate: bool = False,
                        **kwargs) -> DistributedModelSolverResult:
  """Meta analysis.

  Args:
    single_log_likelihood_or_score_fn: a log likelihood function or a score
      function for the single site version of the model.
    use_likelihood: whether ``single_log_likelihood_or_score_fn`` is a
      log likelihood function.
    use_only_dims: a sequence of dimensions to perform the analysis on. Defaults
      to the full dimension.
    univariate: whether to perform univariate meta analysis.
    kwargs: arguments to be passed into the solver.

  Returns:
    a function that takes in the same parameters as
    ``single_log_likelihood_or_score_fn``, and returns the analysis results.
  """

  def solve_fun(*args):
    sol = vmap(
        solve_single(single_log_likelihood_or_score_fn,
                     use_likelihood=use_likelihood,
                     **kwargs))(*args)
    converged = jnp.all(sol.converged, axis=0)
    I_diag_wo_last = -sol.hessian

    if use_only_dims is not None:
      I_diag_wo_last = jnp.take(jnp.take(I_diag_wo_last, use_only_dims, axis=1),
                                use_only_dims,
                                axis=2)
      pt1_guesses = jnp.take(sol.guess, use_only_dims, axis=-1)
    else:
      pt1_guesses = sol.guess

    if univariate:
      wk = 1. / jnp.diagonal(jnp.linalg.inv(I_diag_wo_last), axis1=-2, axis2=-1)
      guess = jnp.einsum("kp,kp->p", wk, pt1_guesses,
                         optimize="optimal") / jnp.sum(wk, axis=0)
    else:
      guess = jnp.linalg.solve(
          jnp.sum(I_diag_wo_last, axis=0),
          jnp.einsum("Kab,Kb->a",
                     I_diag_wo_last,
                     pt1_guesses,
                     optimize='optimal'))
    return DistributedModelSolverResult(pt1=sol,
                                        pt2=MetaAnalysisResult(
                                            guess=guess, converged=converged))

  return solve_fun


def cov_meta_analysis(use_only_dims: Optional[Sequence] = None,
                      univariate=False):
  """Analytically estimates the covariance of meta analysis."""

  def wrapped(sol: DistributedModelSolverResult):
    I_diag_wo_last = -sol.pt1.hessian
    if use_only_dims is not None:
      I_diag_wo_last = jnp.take(jnp.take(I_diag_wo_last, use_only_dims, axis=1),
                                use_only_dims,
                                axis=2)
    if univariate:
      wk = 1. / jnp.diagonal(jnp.linalg.inv(I_diag_wo_last), axis1=-2, axis2=-1)
      X_DIM = sol.pt2.guess.shape[0]
      cov = jnp.zeros((X_DIM, X_DIM), dtype=sol.pt2.guess.dtype)
      cov = jax.ops.index_update(cov, jnp.diag_indices(X_DIM),
                                 1. / jnp.sum(wk, axis=0))
    else:
      cov = jnp.linalg.inv(jnp.sum(I_diag_wo_last, axis=0))
    return cov

  return wrapped


def cov_H():
  """Analytically estimates the covariance of a solver result, by simply
  using the negative inverse hessian."""

  def wrapped(sol: Union[NewtonSolverResult, DistributedModelSolverResult],
              *args):
    del args
    if isinstance(sol, DistributedModelSolverResult):
      sol = sol.pt2
    return scipy.linalg.inv(-sol.hessian)

  return wrapped


def cov_robust(batch_log_likelihood_or_score_fn,
               num_single_args: int = 1,
               use_likelihood=True,
               sum_group_first=False):
  """Analytically estimates covariance using robust sandwich estimate.

    See [Huber 1967].
  """
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
    if sum_group_first:
      batch_score = jnp.sum(batch_score, axis=1)
    batch_score = batch_score.reshape((-1, batch_score.shape[-1]))
    B = jnp.einsum("ni,nj->ij", batch_score, batch_score, optimize="optimal")
    return A_inv @ B @ A_inv.T

  return wrapped


def cov_group_correction(
    batch_single_log_likelihood_or_score_fn: Callable,
    batch_distributed_log_likelihood_or_score_fn: Callable,
    distributed_cross_hessian_fn: Optional[Callable] = None,
    num_single_args: int = 1,
    batch_single_use_likelihood=True,
    batch_distributed_use_likelihood=True,
    robust=False,
    robust_sum_group_first=False):
  """Computes covariance with grouped correction."""
  del robust_sum_group_first  # doesn't work right now

  if batch_single_use_likelihood:
    batch_single_score_fn = jacfwd(batch_single_log_likelihood_or_score_fn,
                                   num_single_args - 1)
  else:
    batch_single_score_fn = batch_single_log_likelihood_or_score_fn

  if batch_distributed_use_likelihood:
    batch_distributed_score_fn = jacfwd(
        batch_distributed_log_likelihood_or_score_fn, num_single_args - 1)
    if distributed_cross_hessian_fn is None:
      distributed_cross_hessian_fn = jacfwd(
          jacrev(sum_loglik(batch_distributed_log_likelihood_or_score_fn),
                 num_single_args - 1), -1)
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

    I_diag_inv_wo_last = jnp.linalg.inv(I_diag_wo_last)
    I_diag_inv_last = jnp.linalg.inv(I_diag_last)

    if robust:
      pt1_batch_scores = vmap(batch_single_score_fn)(*distributed_args)
      pt2_batch_scores = batch_distributed_score_fn(*args)  # batch_dims x X_dim

      B_diag_wo_last = jnp.einsum("kbi,kbj->kij",
                                  pt1_batch_scores,
                                  pt1_batch_scores,
                                  optimize="optimal")
      B_diag_last = jnp.einsum("ksi,ksj->ij",
                               pt2_batch_scores,
                               pt2_batch_scores,
                               optimize="optimal")
      B_row_wo_last = jnp.einsum("kbi,kbj->kij",
                                 pt2_batch_scores,
                                 pt1_batch_scores,
                                 optimize="optimal")

      S = jnp.einsum("ab,bBc,Bcd->Bad",
                     I_diag_inv_last,
                     I_row_wo_last,
                     I_diag_inv_wo_last,
                     optimize="optimal")

      sas = jnp.einsum("Bab,Bbc,Bdc->ad",
                       S,
                       B_diag_wo_last,
                       S,
                       optimize="optimal")
      sb1s = jnp.einsum("ab,Bbc,Bdc->ad",
                        I_diag_inv_last,
                        B_row_wo_last,
                        S,
                        optimize="optimal")
      sb2s = sb1s.T  # pylint: disable=no-member
      scs = jnp.einsum('ab,bc,dc->ad',
                       I_diag_inv_last,
                       B_diag_last,
                       I_diag_inv_last,
                       optimize='optimal')
      cov = sas - sb1s - sb2s + scs

    else:

      S = jnp.einsum("ab,bBc->Bac",
                     I_diag_inv_last,
                     I_row_wo_last,
                     optimize="optimal")

      cov = jnp.einsum(
          "Bab,Bbc,Bdc->ad", S, I_diag_inv_wo_last, S,
          optimize='optimal') + I_diag_inv_last

    return cov

  return wrapped
