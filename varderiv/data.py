"""Data generation."""

import math
import functools

import numpy as onp

import jax.numpy as np
from jax import vmap
import jax.lax
import jax.config

from jax import random as jrandom

if hasattr(jrandom, "permutation"):
  jrandom_shuffle = jrandom.permutation
else:
  # backward compat
  jrandom_shuffle = jrandom.shuffle

if jax.config.read("jax_enable_x64"):
  floatt = np.float64
else:
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
  bernoulli_thetas = np.array([0.5, 0.3, 0.7])
  normal_variances = np.array([1, 0.5, 1.5])

  bernoulli_theta = bernoulli_thetas[group_label]
  normal_variance = normal_variances[group_label]

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


def T_star_factors_gamma_gen(shape, scale):

  def wrapped(key, K):
    return 1. / (jrandom.gamma(key, a=shape, shape=(K,)) * scale)

  return wrapped


@functools.lru_cache(maxsize=None)
def data_generator(N,
                   X_dim,
                   group_sizes,
                   T_star_factors=None,
                   X_generator=None,
                   exp_scale=3.5,
                   return_T=False,
                   return_T_star=False):
  """HOF for data generation.

  The function is cached so that we avoid potential repeating jits'.
  """

  ret_signature = "(N,p),(N),(p),(N)"
  if return_T:
    ret_signature = "(N)," + ret_signature
  if return_T_star:
    ret_signature = "(N)," + ret_signature

  wrapped_signature = "(k)->" + ret_signature

  K = len(group_sizes)

  if T_star_factors is None:
    T_star_factors = tuple([1] * K)

  if isinstance(T_star_factors, tuple):
    assert len(T_star_factors) == K
    random_T_star = False
  else:
    assert callable(T_star_factors)
    random_T_star = True

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
    # beta = np.array([-1, 0, 1], dtype=np.float32)
    beta = np.arange(1, X_dim + 1, dtype=floatt) / X_dim

    key, *subkeys = jrandom.split(key, K + 1)
    subkeys = np.stack(subkeys)

    X = np.zeros((N, X_dim), dtype=floatt)
    max_group_size = max(group_sizes)

    def gen_X(carry, group_size):
      X, group_label, cur_idx = carry
      X_group = X_generator(max_group_size,
                            X_dim,
                            subkeys[group_label],
                            group_label=group_label)
      X = jax.lax.dynamic_update_slice(X, X_group,
                                       np.array([cur_idx, 0], dtype=np.int32))
      return (X, group_label + 1, cur_idx + group_size), 0

    (X, _, _), _ = jax.lax.scan(gen_X, (X, 0, 0), np.array(group_sizes))

    group_labels = np.repeat(np.arange(K), group_sizes)

    key, subkey = jrandom.split(key)
    u = jrandom.uniform(subkey, shape=(N,), minval=0, maxval=1)
    if random_T_star:
      key, subkey = jrandom.split(key)
      T_star_factors_ = T_star_factors(subkey, K)
    else:
      T_star_factors_ = np.array(T_star_factors)
    T_star_factors_per_item = np.repeat(T_star_factors_, group_sizes)
    T_star = -T_star_factors_per_item * np.log(u) / np.exp(X.dot(beta))

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
    ret = (X, delta, beta, group_labels)
    if return_T:
      ret = (T,) + ret
    if return_T_star:
      T_star = np.take(T_star, sorted_idx, axis=0)
      ret = (T_star,) + ret
    return ret

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


def group_data_by_labels(group_labels, *data, K=1, group_size=-1):
  """Given data group indices, compute groupped data by padding.
  #TODO update docs
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
  if group_size < 0:
    group_size = np.max(np.vectorize(functools.partial(np.bincount, length=K),
                                     signature="(N)->(K)")(group_labels),
                        axis=-1)

  return tuple(
      group_by_labels(group_labels, d, K=K, group_size=group_size)
      for d in data)


def _group_by_labels(group_labels, X, K=1, group_size=-1):
  """A convenience function for groupping X by labels in Jax."""
  X_grouped = np.zeros((K, group_size) + X.shape[1:], dtype=X.dtype)
  group_cnts = np.zeros((K,), np.int32)

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
    X: array of dimension (...batch_dims... , N, ...X_dims...)
    group_labels: array of dimension (...batch_dims..., N)

  Returns:
    X_grouped: dimension (...batch_dims..., K, group_size, ...X_dims...)
  """
  batch_dim = len(group_labels.shape) - 1
  X_dim = len(X.shape) - batch_dim - 1

  assert X.shape[:batch_dim + 1] == group_labels.shape

  fun = functools.partial(_group_by_labels, K=K, group_size=group_size)
  for _ in range(X_dim):
    fun = vmap(fun, in_axes=(None, batch_dim + 1), out_axes=batch_dim + 2)
  for _ in range(batch_dim):
    fun = vmap(fun, in_axes=0, out_axes=0)
  return fun(group_labels, X)


key = jrandom.PRNGKey(0)
key, data_generation_key = jrandom.split(key)


@functools.partial(np.vectorize, signature="(N,p),(p)->(N,p),(p),(p)")
def normalize(X, beta):
  X = X - np.mean(X, axis=0)
  scale = X.shape[0] / np.linalg.norm(X, ord=1, axis=0)
  X *= scale
  beta /= scale
  return X, beta, scale
