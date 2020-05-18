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

  InternalState = collections.namedtuple("InternalState", "guess value step last_valid_u halving")

  def newton_update(state: InternalState):
    
    def do_halving(state):
      guess, value, step, last_valid_u, halving = state
      new_halving = halving * 2
      guess = (guess - last_valid_u / halving) + last_valid_u / new_halving
      value = fn(guess)
      return InternalState(guess, value, step + 1, last_valid_u, new_halving)
    
    def do_normal_update(args):
      u, new_guess, new_value, step = args
      return InternalState(new_guess, new_value, step + 1, u, 1)

    guess, value, step, last_valid_u, halving = state
    jacobian = jac_fn(guess)
    u = scipy.linalg.solve(-jacobian, value, sym_pos=sym_pos)

    new_guess = state.guess + u
    new_value = fn(new_guess)

    is_finite = np.logical_and(np.all(np.isfinite(u)), np.all(np.isfinite(new_value)))

    return jax.lax.cond(
      is_finite, 
      (u, new_guess, new_value, state.step),
      do_normal_update,
      state,
      do_halving
    )

  def stop_cond(state):
    return np.logical_and(
        state.step < max_num_steps,
        np.logical_or(state.halving > 1, np.linalg.norm(state.value, ord=np.inf) > norm_stop_thres))

  initial_value = fn(initial_guess)
  initial_state = InternalState(initial_guess, initial_value, 0, np.zeros_like(initial_guess), 1)
  ret = jax.lax.while_loop(stop_cond, newton_update, initial_state)

  return NewtonSolverState(ret.guess, ret.value, ret.step)


solve_newton_jit = jit(solve_newton)
