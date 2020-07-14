"""Newton solver."""

import collections

import jax.lax
import jax.numpy as np
import jax.scipy as scipy

from varderiv.generic.hess import value_jac_and_hessian

NewtonSolverResult = collections.namedtuple(
    "NewtonSolverResult", "guess loglik score hessian step converged")


def do_halving(args):
  state, _ = args
  halving_ = state.halving + 1
  new_guess_ = ((state.new_guess + halving_ * state.guess) / (halving_ + 1.0))
  return state._replace(new_guess=new_guess_, halving=halving_)


def do_normal_update(args):
  state, (new_loglik, new_score, _, cho_factor, _) = args
  u = scipy.linalg.cho_solve((cho_factor, False), new_score)
  new_guess_ = state.new_guess + u
  return state._replace(guess=state.new_guess,
                        new_guess=new_guess_,
                        loglik=new_loglik,
                        halving=0)


def do_converged(args):
  state, (new_loglik, new_score, new_hessian, _, _) = args
  return state._replace(loglik=new_loglik,
                        score=new_score,
                        hessian=new_hessian,
                        converged=True)


def do_work(args):
  is_finite = args[1][-1]
  state = jax.lax.cond(is_finite, do_normal_update, do_halving, operand=args)
  return state._replace(step=state.step + 1)


def solve_newton(likelihood_fn, initial_guess, eps=1e-6,
                 max_num_steps=10) -> NewtonSolverResult:
  """HOF for newton's method solver."""

  value_jac_and_hessian_fn = value_jac_and_hessian(likelihood_fn)

  X_DIM = initial_guess.shape[0]

  InternalState = collections.namedtuple(
      "InternalState",
      "guess new_guess loglik score hessian step halving converged")

  def newton_update(state: InternalState):
    _, new_guess, loglik, _, _, _, _, _ = state
    new_loglik, new_score, new_hessian = value_jac_and_hessian_fn(new_guess)
    cho_factor = scipy.linalg.cholesky(-new_hessian, lower=False)

    is_finite = np.logical_and(
        np.all(np.isfinite(new_loglik)),
        np.logical_and(np.all(np.isfinite(new_score)),
                       np.all(np.isfinite(cho_factor))))
    converged = np.logical_and(is_finite,
                               np.allclose(new_loglik, loglik, rtol=eps))

    state = state._replace(converged=converged)
    args = (state, (new_loglik, new_score, new_hessian, cho_factor, is_finite))
    return jax.lax.cond(converged, do_converged, do_work, operand=args)

  def loop_cond(state):
    return np.logical_and(state.step < max_num_steps,
                          np.logical_not(state.converged))

  initial_state = InternalState(initial_guess, initial_guess, np.inf,
                                np.zeros_like(initial_guess),
                                np.zeros((X_DIM, X_DIM)), 0, 0, False)
  state = jax.lax.while_loop(loop_cond, newton_update, initial_state)

  def do_recover_last(_):
    loglik, score, hessian = value_jac_and_hessian_fn(state.guess)
    return state._replace(loglik=loglik, score=score, hessian=hessian)

  state = jax.lax.cond(state.converged,
                       lambda _: state,
                       do_recover_last,
                       operand=None)

  return NewtonSolverResult(state.guess, state.loglik, state.score,
                            state.hessian, state.step, state.converged)


if __name__ == "__main__":

  def test_fun(x):
    return np.squeeze(x**3 + x**2 - x + np.cos(x))

  print(solve_newton(test_fun, np.array([1.])))
