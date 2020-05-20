"""Data generation."""

import math
import functools

import numpy as onp

import jax.numpy as np
from jax import vmap
import jax.lax

from jax import random as jrandom

if hasattr(jrandom, "permutation"):
  jrandom_shuffle = jrandom.permutation
else:
  # backward compat
  jrandom_shuffle = jrandom.shuffle

floatt = np.float32

# pylint: disable=redefined-outer-name


def default_Xi_generator(N, dim, key, group_label=0):
  del group_label
  return jax.lax.cond(
      dim % 2 == 0,
      key, \
      lambda key: jrandom.bernoulli(key, p=0.5, shape=(N,)).astype(floatt),
      key, \
      lambda key: jrandom.normal(key, shape=(N,)))


def grouping_Xi_generator(N, dim, key, group_label=0):
  if group_label == 0:
    bernoulli_theta, normal_variance = 0.5, 1
  elif group_label == 1:
    bernoulli_theta, normal_variance = 0.3, 0.5
  elif group_label == 2:
    bernoulli_theta, normal_variance = 0.7, 1.5
  return jax.lax.cond(
      dim % 2 == 0,
      key, \
      lambda key: jrandom.bernoulli(key, p=bernoulli_theta,
                                    shape=(N,)).astype(floatt),
      key, \
      lambda key: jrandom.normal(key, shape=(N,))) * normal_variance


def X_group_generator_indep_dim(N,
                                X_dim,
                                key,
                                group_label=0,
                                Xi_generator=default_Xi_generator):
  """Helper utility that lifts a generator that produces independent Xi."""
  gen_X_fn = functools.partial(Xi_generator, N, group_label=group_label)
  dims = np.arange(X_dim, dtype=np.int32)
  subkeys = jrandom.split(key, X_dim)
  return vmap(gen_X_fn, (0, 0), 1)(dims, subkeys)


default_X_generator = functools.partial(X_group_generator_indep_dim,
                                        Xi_generator=default_Xi_generator)
grouping_X_generator = functools.partial(X_group_generator_indep_dim,
                                         Xi_generator=grouping_Xi_generator)


@functools.lru_cache(maxsize=None)
def data_generator(N,
                   X_dim,
                   group_sizes,
                   X_generator=None,
                   exp_scale=3.5,
                   return_T=False):
  """HOF for data generation.

  The function is cached so that we avoid potential repeating jits'.
  """

  if return_T:
    wrapped_signature = "(k)->(N),(N,p),(N),(p),(N)"
  else:
    wrapped_signature = "(k)->(N,p),(N),(p),(N)"

  K = len(group_sizes)

  if X_generator is None:
    X_generator = default_X_generator

  @functools.partial(np.vectorize, signature=wrapped_signature)
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
    beta = np.array([5, 6, 7], dtype=np.float32)

    key, *subkeys = jrandom.split(key, K + 1)
    subkeys = np.stack(subkeys)

    Xs = []
    idx = 0
    for group_label, group_size in enumerate(group_sizes):
      Xs.append(
          X_generator(group_size,
                      X_dim,
                      subkeys[group_label],
                      group_label=group_label))
      idx += group_size
    X = np.concatenate(Xs)

    group_labels = np.repeat(np.arange(K), group_sizes)

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
    group_labels = np.take(group_labels, sorted_idx, axis=0)

    # X = X - np.mean(X, axis=0)
    if return_T:  # pylint: disable=no-else-return
      return T, X, delta, beta, group_labels
    else:
      return X, delta, beta, group_labels

  return wrapped


@functools.lru_cache(maxsize=None)
def group_sizes_generator(N, K, group_labels_generator_kind="random", **kwargs):
  """HOF for group sizes generation."""

  # Validate arguments
  if group_labels_generator_kind == "random":
    assert kwargs == {}
  elif group_labels_generator_kind == "same":
    assert kwargs == {}
  elif group_labels_generator_kind == "arithmetic_sequence":
    assert kwargs.keys() == {"start_val"}
  elif group_labels_generator_kind == "single_ladder":
    assert kwargs.keys() == {"start_val", "repeat_start"}
  elif group_labels_generator_kind == "manual":
    assert kwargs.keys() == {"sizes"}
    assert len(kwargs["sizes"]) == K and sum(kwargs["sizes"]) == N

  if group_labels_generator_kind == "same":
    group_sizes = onp.repeat(N // K, K)
    group_sizes[:N % K] += 1

  elif group_labels_generator_kind == "arithmetic_sequence":
    start_val = kwargs["start_val"]
    step = int(math.floor(2 * (N - start_val * K) / ((K - 1) * K)))
    end_val = start_val + (K - 1) * step
    group_sizes = onp.arange(start_val, end_val + 1, step)
    current_total = (start_val + end_val) * K // 2
    residual = N - current_total
    group_sizes[K - residual:] += 1

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
    group_sizes[K - residual:] += 1

  elif group_labels_generator_kind == "manual":
    group_sizes = kwargs["sizes"]

  else:
    raise TypeError("Invalid group_label_generator_kind")

  return tuple(group_sizes)


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


def _pad_X_delta(X, delta, indices, padded_group_size):
  """Currently Unused."""
  X_group = onp.take(X, indices, axis=0)
  X_group = onp.pad(X_group, [(0, padded_group_size - X_group.shape[0]),
                              (0, 0)])

  delta_group = onp.take(delta, indices, axis=0)
  delta_group = onp.pad(delta_group, (
      0,
      padded_group_size - delta_group.shape[0],
  ))
  return X_group, delta_group


def group_data_by_labels(batch_size, K, X, delta, group_labels):
  """Given data group indices, compute groupped data by padding.

  Args:
    - batch_size: length of a batch of X
    - K: number of groups
    - X: array of shape (batch_size, N, P).
    - delta: array of shape (batch_size, N)
    - group_labels: array of shape (batch_size, N)

  Returns:
    tuple of X_groups, delta_groups
    - X_groups: array of shape (batch_size, K, group_size, P)
    - delta_groups: array of shape (batch_size, K, group_size)
  """
  X = onp.array(X)
  delta = onp.array(delta)
  group_labels = onp.array(group_labels)

  batch_mode = True
  if batch_size <= 1 and len(X.shape) == 2:
    batch_mode = False
    X = X.reshape((1,) + X.shape)
    delta = delta.reshape((1,) + delta.shape)
    group_labels = group_labels.reshape((1,) + group_labels.shape)

  batch_size = X.shape[0]
  # X_dim = X.shape[-1]

  group_mask = onp.array(
      [[group_labels[i] == k for k in range(K)] for i in range(batch_size)])

  padded_group_size = onp.max(onp.sum(group_mask, axis=(-1,)))
  padded_group_size = int(math.ceil(padded_group_size / 10)) * 10

  all_X_groups = [None] * batch_size
  all_delta_groups = [None] * batch_size
  for i in range(batch_size):
    X_groups = [None] * K
    delta_groups = [None] * K
    for k in range(K):
      mask = group_mask[i][k]
      X_group = X[i, mask]
      X_group = onp.pad(X_group, [(0, padded_group_size - X_group.shape[0]),
                                  (0, 0)])
      delta_group = delta[i, mask]
      delta_group = onp.pad(delta_group, (
          0,
          padded_group_size - delta_group.shape[0],
      ))
      X_groups[k] = X_group
      delta_groups[k] = delta_group
    all_X_groups[i] = X_groups
    all_delta_groups[i] = delta_groups

  all_X_groups = np.array(all_X_groups)
  all_delta_groups = np.array(all_delta_groups)

  if not batch_mode:
    all_X_groups = all_X_groups.reshape(all_X_groups.shape[1:])
    all_delta_groups = all_delta_groups.reshape(all_delta_groups.shape[1:])

  return all_X_groups, all_delta_groups


key = jrandom.PRNGKey(0)
key, data_generation_key = jrandom.split(key)


def normalize(X, beta):
  X = X - onp.mean(X, axis=0)
  scale = X.shape[0] / onp.linalg.norm(X, ord=1, axis=0)
  X *= scale
  beta /= scale
  return X, beta, scale
