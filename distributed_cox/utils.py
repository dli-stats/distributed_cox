"""General utilities."""

import functools

import jax.numpy as jnp
import jax.ops
import jax.lax
import jax.random as jrandom


def _group_by_labels(group_labels, X, K=1, group_size=-1):
  """A convenience function for groupping X by labels in Jax."""
  X_grouped = jnp.zeros((K, group_size) + X.shape[1:], dtype=X.dtype)
  group_cnts = jnp.zeros((K,), jnp.int32)

  def group_by(data, var):
    x, g = var
    x_grouped, group_cnts = data
    # append entries into specified group
    x_grouped = jax.ops.index_update(x_grouped, (g, group_cnts[g]), x)
    # track how many entries appended into each group
    group_cnts = jax.ops.index_add(group_cnts, g, 1)
    return (x_grouped, group_cnts), 0  # '0' is just a dummy value

  (X_grouped, group_cnts), _ = jax.lax.scan(
      group_by,
      (X_grouped, group_cnts),  # initial state
      (X, group_labels))  # data to loop over

  return X_grouped


def group_by_labels(group_labels, X, K: int = 1, group_size: int = -1):
  """Group data by labels.

  Args:
    X: array of dimension ``(...batch_dims... , N, ...X_dims...)``.
    group_labels: array of dimension ``(...batch_dims..., N)``.

  Returns:
    array of dimension ``(...batch_dims..., K, group_size, ...X_dims...)``.
  """
  batch_dim = len(group_labels.shape) - 1

  assert X.shape[:batch_dim + 1] == group_labels.shape

  fun = functools.partial(_group_by_labels, K=K, group_size=group_size)
  # for _ in range(batch_dim):
  #   fun = vmap(fun, in_axes=0, out_axes=0)
  return fun(group_labels, X)


# Distributions extending JAX's jnp.random


def bernoulli(key, theta, shape=None, dtype=bool):
  return jrandom.bernoulli(key, p=theta, shape=shape).astype(dtype)


def normal(key, mean, std, shape=None, dtype=jnp.float64):
  return jrandom.normal(key, shape=shape, dtype=dtype) * std + mean


def gamma(key, a, scale, shape=None, dtype=jnp.float64):
  return jrandom.gamma(key, a=a, shape=shape, dtype=dtype) * scale


def exponential(key, scale: float = 3.5, shape=None, dtype=jnp.float64):
  return jrandom.exponential(key, shape=shape, dtype=dtype) * scale
