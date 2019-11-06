"""Newton solver."""

import collections

import jax.lax
import jax.numpy as np
from jax import jacrev, jacfwd, jit
import jax.scipy as scipy

NewtonSolverState = collections.namedtuple("NewtonSolverState",
                                           "guess value step")


def solve_newton(fn,
                 initial_guess,
                 sym_pos=False,
                 norm_stop_thres=1e-3,
                 jac_mode='forward',
                 max_num_steps=10):
  """HOF for newton's method solver."""

  if jac_mode == 'forward':
    jac_fn = jacfwd(fn, 0)
  else:
    jac_fn = jacrev(fn, 0)

  initial_value = fn(initial_guess)

  InternalState = collections.namedtuple("InternalState", "guess value step")

  def newton_update(state: InternalState):
    guess, value, step = state
    jacobian = jac_fn(guess)
    guess = guess + scipy.linalg.solve(-jacobian, value, sym_pos=sym_pos)
    value = fn(guess)
    return InternalState(guess, value, step + 1)

  def stop_cond(state):
    return np.logical_and(
        state.step < max_num_steps,
        np.linalg.norm(state.value, ord=np.inf) > norm_stop_thres)

  ret = jax.lax.while_loop(stop_cond, newton_update,
                           InternalState(initial_guess, initial_value, 0))

  return NewtonSolverState(ret.guess, ret.value, ret.step)


solve_newton_jit = jit(solve_newton)
