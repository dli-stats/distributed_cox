"""Data generation."""

import math
import functools

import numpy as onp

import jax.numpy as np
from jax import jit
from jax.experimental.vectorize import vectorize

from jax import random as jrandom

floatt = np.float32

# pylint: disable=redefined-outer-name


def default_X_generator(key, dim, N):
  if dim % 2 == 0:  # pylint: disable=no-else-return
    return jrandom.bernoulli(key, p=0.5, shape=(N,)).astype(floatt)
  elif dim % 2 == 1:
    return jrandom.normal(key, shape=(N,))


@functools.lru_cache(maxsize=None)
def data_generator(N, X_dim, X_generator=default_X_generator, exp_scale=3.5):
  """HOF for data generation.

  The function is cached so that we avoid potential repeating jits'.
  """

  @vectorize("(k)->(N,p),(N),(p)")
  def wrapped(key):
    r"""Generates dummy data.

    The generative process works as follow:
      1. Sample beta
      2. Sample X with Xr
      3. Sample T^* for each X
      4. Sample C according to `\frac{\log(u)}{\exp(X \cdot \beta)}`
          where `u \sim \text{Unif}(0, 1)`
      5. Reorder X by `T = \min(T^*, C)`
    """
    beta = np.arange(1, X_dim + 1, dtype=floatt) / X_dim
    X = []
    for dim in range(X_dim):
      key, subkey = jrandom.split(key)
      X.append(X_generator(subkey, dim, N))
    X = np.dstack(X)
    X = X.reshape((X.shape[1:]))

    key, subkey = jrandom.split(key)
    u = jrandom.uniform(subkey, shape=(N,), minval=0, maxval=1)
    T_star = -np.log(u) / np.exp(X.dot(beta))

    key, subkey = jrandom.split(key)
    C = jrandom.exponential(subkey, shape=(N,)) * exp_scale
    delta = T_star <= C

    T = np.minimum(T_star, C)

    sorted_idx = np.argsort(-T)  # sort T descending

    T = np.take(T, sorted_idx, axis=0)
    X = np.take(X, sorted_idx, axis=0)
    delta = np.take(delta, sorted_idx, axis=0)
    return X, delta, beta

  return wrapped


@functools.lru_cache(maxsize=None)
def group_labels_generator(N, K, group_labels_generator_kind="random",
                           **kwargs):
  """HOF for group labels generation."""

  # Validate arguments
  if group_labels_generator_kind in {"random", "same"}:
    assert kwargs == {}
  elif group_labels_generator_kind == "arithmetic_sequence":
    assert kwargs.keys() == {"start_val"}
  elif group_labels_generator_kind == "single_ladder":
    assert kwargs.keys() == {"start_val", "repeat_start"}

  @vectorize("(k)->(N)")
  def generate_group_labels(key):
    if group_labels_generator_kind == "random":  # pylint: disable=no-else-return
      key, subkey = jrandom.split(key)
      group_labels = jrandom.randint(subkey, (N,), minval=0, maxval=K)
      return group_labels
    elif group_labels_generator_kind == "same":
      group_labels = np.arange(N) % K
      group_labels = jrandom.shuffle(key, group_labels)
      return group_labels
    elif group_labels_generator_kind == "arithmetic_sequence":
      start_val = kwargs["start_val"]
      step = int(math.floor(2 * (N - start_val * K) / ((K - 1) * K)))
      end_val = start_val + (K - 1) * step
      group_sizes = onp.arange(start_val, end_val + 1, step)
      current_total = (start_val + end_val) * K // 2
      residual = N - current_total
      group_sizes[-residual:] += 1
      group_labels = onp.repeat(onp.arange(K), group_sizes)
      group_labels = jrandom.shuffle(key, group_labels)
      return group_labels
    elif group_labels_generator_kind == "single_ladder":
      start_val = kwargs["start_val"]
      repeat_start = kwargs["repeat_start"]
      assert 1 <= repeat_start <= K - 1
      rest_val = int(
          math.floor((N - start_val * repeat_start) / (K - repeat_start)))
      group_sizes = onp.array([start_val] * repeat_start + [rest_val] *
                              (K - repeat_start))
      current_total = onp.sum(group_sizes)
      residual = N - current_total
      group_sizes[-residual:] += 1
      group_labels = onp.repeat(onp.arange(K), group_sizes)
      group_labels = jrandom.shuffle(key, group_labels)
      return group_labels
    else:
      raise TypeError("Invalid group_label_generator_kind")

  return generate_group_labels


def group_labels_to_indices(K, group_labels):
  """Computes group indices from group_labels.

  Note: this function is done on CPU since JAX's current support on indexing.
  Therefore this WILL incurr extra host-device transfer
  if group_labels is on GPU.
  """
  group_labels = onp.array(group_labels)
  batch_mode = True
  if len(group_labels.shape) == 1:
    batch_mode = False
    group_labels = group_labels.reshape((1, -1))

  assert len(group_labels.shape) == 2
  #TODO(camyang) make this more efficient?
  ret = [[onp.flatnonzero(gl == k) for k in range(K)] for gl in group_labels]
  if not batch_mode:
    return ret[0]
  return ret


@functools.partial(jit, static_argnums=(0, 1))
def group_data_by_labels(batch_size, K, X, delta, group_indices):
  """Given data group indices, compute groupped data by padding.

  Args:
    - batch_size: length of a batch of X
    - K: number of groups
    - X: array of shape (batch_size, N, P).
    - delta: array of shape (batch_size, N)
    - group_indices: nested list of shape (batch_size, K); each list element is
      a flattened array representing indices in a group

  Returns:
    tuple of X_groups, delta_groups
    - X_groups: array of shape (batch_size, K, group_size, P)
    - delta_groups: array of shape (batch_size, K, group_size)
  """
  batch_mode = True
  if batch_size <= 1 and len(X.shape) == 2:
    batch_mode = False
    X = X.reshape((1,) + X.shape)
    delta = delta.reshape((1,) + delta.shape)
    group_indices = [group_indices]

  batch_size = X.shape[0]
  padded_group_size = max(
      *[len(group_indices[i][k]) for k in range(K) for i in range(batch_size)])

  all_X_groups, all_delta_groups = [], []
  for i in range(batch_size):
    X_groups, delta_groups = [], []
    for k in range(K):
      group_idxs = group_indices[i][k]

      X_group = np.take(X[i], group_idxs, axis=0)
      X_group = np.pad(X_group, [(0, padded_group_size - X_group.shape[0]),
                                 (0, 0)])

      delta_group = np.take(delta[i], group_idxs, axis=0)
      delta_group = np.pad(delta_group, (
          0,
          padded_group_size - delta_group.shape[0],
      ))

      X_groups.append(X_group)
      delta_groups.append(delta_group)
    all_X_groups.append(X_groups)
    all_delta_groups.append(delta_groups)

  all_X_groups = np.array(all_X_groups)
  all_delta_groups = np.array(all_delta_groups)

  if not batch_mode:
    all_X_groups = all_X_groups.reshape(all_X_groups.shape[1:])
    all_delta_groups = all_delta_groups.reshape(all_delta_groups.shape[1:])

  return all_X_groups, all_delta_groups


key = jrandom.PRNGKey(0)
key, data_generation_key = jrandom.split(key)
