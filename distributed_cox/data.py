"""Data generation of simulated data for the Cox model."""

from typing import Sequence, Tuple, Optional

import math
import functools

import simpleeval as se

import numpy as onp

import jax.numpy as jnp
from jax import vmap
import jax.lax
import jax.config

from jax import random as jrandom

import distributed_cox.utils as vutils

if hasattr(jrandom, "permutation"):
  jrandom_shuffle = jrandom.permutation
else:
  # backward compat
  jrandom_shuffle = jrandom.shuffle

if jax.config.read("jax_enable_x64"):
  floatt = jnp.float64
else:
  floatt = jnp.float32

# pylint: disable=redefined-outer-name
# pylint: disable=invalid-unary-operand-type


def bernoulli(theta):

  def wrapped(key, shape=None):
    return vutils.bernoulli(key, theta=theta, shape=shape, dtype=floatt)

  return wrapped


def normal(mean, std):

  def wrapped(key, shape=None):
    return vutils.normal(key, mean=mean, std=std, shape=shape, dtype=floatt)

  return wrapped


def _grouping_Xi_generator_any_K(N, dim, key, group_label=0, g_dists=None):
  """Helper that generates X in a single group."""

  def group_fun(dists):

    def wrapped(key):
      return jax.lax.switch(dim % len(dists),
                            [functools.partial(d, shape=(N,)) for d in dists],
                            key)

    return wrapped

  return jax.lax.switch(group_label % len(g_dists),
                        [group_fun(gd) for gd in g_dists], key)


def make_X_generator(N,
                     X_dim,
                     key,
                     group_label=0,
                     g_dists=None,
                     correlated_dims: Optional[Sequence[Tuple]] = None,
                     correlated_weights: Optional[Sequence[float]] = None):
  """Helper utility to generatoe each group.

  We generage each ``X[group_label]`` according to::

    for i in range(X_DIM):
      g_dist = g_dists[group_label % len(g_dists)]
      X[group_label, i] ~ g_dist[i % len(g_dist)]

    correlated_from, correlated_to = correlated_dims[
                                      group_label % len(correlated_dims)]
    correlated_weight = correlated_weights[
                                      group_label % len(correlated_weights)]
    X[group_label, correlated_to] += X[group_label, correlated_from]

  Args:
    N: size of the group. Note that in a multi-group setting, this should
      usually be the max size of all groups.
    X_dim: number of covariate dimensions of ``X``.
    key: use-once random key.
    group_label: the group label for generating this group.
    g_dists: a nested list of distributions.
    correlated_dims: sequence of tuple of pairs of correlated indices
    correlated_weights: weights for the correlations.

  Returns:
    array of shape ``(N, X_dim)``.
  """
  gen_X_fn = functools.partial(_grouping_Xi_generator_any_K,
                               N,
                               group_label=group_label,
                               g_dists=g_dists)
  dims = jnp.arange(X_dim, dtype=jnp.int32)
  subkeys = jrandom.split(key, X_dim)
  X = vmap(gen_X_fn, (0, 0), 1)(dims, subkeys)
  if correlated_dims is None:
    return X

  correlated_dims = jnp.array(correlated_dims, dtype=jnp.int32)
  correlated_weights = jnp.array(correlated_weights, dtype=floatt)
  correlated_from, correlated_to = correlated_dims[group_label %
                                                   correlated_dims.shape[0]]
  weight = correlated_weights[group_label % correlated_weights.shape[0]]
  X = jax.ops.index_add(X, jax.ops.index[:, correlated_to],
                        weight * X[:, correlated_from])
  return X


default_g_dists = [[bernoulli(0.5), normal(0, 1.)]]

default_g_dists2 = [[bernoulli(0.5), normal(0, 1.)],
                    [bernoulli(0.3), normal(0, 0.25)],
                    [bernoulli(0.7), normal(0, 2.25)]]

default_g_dists3 = [[normal(0, 1.),
                     normal(0, 0.04),
                     bernoulli(0.5)],
                    [bernoulli(0.1),
                     normal(0, 0.5),
                     normal(2, 0.5)],
                    [bernoulli(0.9),
                     normal(0, 0.04),
                     normal(-1, 1.5)]]

default_X_generator = functools.partial(make_X_generator,
                                        g_dists=default_g_dists)
grouping_X_generator_K3 = functools.partial(make_X_generator,
                                            g_dists=default_g_dists2)
correlated_X_generator = functools.partial(
    make_X_generator,
    g_dists=[[bernoulli(0.5), normal(0, 1.),
              normal(0, 0.01)]],
    correlated_dims=[[-1, -2]],
    correlated_weights=[1.])


def T_star_factors_inv_gamma_gen(shape, scale):
  """Generates ``T_start_factors`` according the inverse of gamma distribution.

  Args:
    shape, scale: the parameters of the gamma distribution.

  Returns:
    A function that generates the inverse of the samples from the specified
    gamma distribution. This function is suitable to be passed into
    :py:func:`data_generator` directly as the parameter `T_star_factors`.
  """

  def wrapped(key, K):
    return 1. / vutils.gamma(
        key, a=shape, scale=scale, shape=(K,), dtype=floatt)

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
  """Higher order function for data generation.

  Args:
    N: total data points.
    X_dim: number of covariate dimensions of ``X``.
    group_sizes: the group sizes for each group. Must be ordered (lower first).
    T_star_factors: a multiplicative factor that is applied to each item.
      Defaults to 1 for all items (no scaling).
    X_generator: a callable that generates each group of ``X``. See
      :py:func:`make_X_generator` for more details.
    exp_scale: the exponential scale.
    return_T: whether to return `T`.
    return_T: whether to return `T_star`.

  Returns:
    a function that takes in a random key, and returns the generated data using
    that key. The generated data is a tuple
    ``(X, delta, beta, group_labels, [T, [T_star]])``.
  """
  assert (sorted(group_sizes) == list(group_sizes)
         ), "Group sizes must be increasing"

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

  @functools.partial(jnp.vectorize, signature=wrapped_signature)
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
    beta = jnp.arange(1, X_dim + 1, dtype=floatt) / X_dim

    key, *subkeys = jrandom.split(key, K + 1)
    subkeys = jnp.stack(subkeys)

    X = jnp.zeros((N, X_dim), dtype=floatt)
    max_group_size = max(group_sizes)

    def gen_X(carry, group_size):
      X, group_label, cur_idx = carry
      X_group = X_generator(max_group_size,
                            X_dim,
                            subkeys[group_label],
                            group_label=group_label)
      X = jax.lax.dynamic_update_slice(X, X_group,
                                       jnp.array([cur_idx, 0], dtype=jnp.int32))
      return (X, group_label + 1, cur_idx + group_size), 0

    (X, _, _), _ = jax.lax.scan(gen_X, (X, 0, 0), jnp.array(group_sizes))

    group_labels = jnp.repeat(jnp.arange(K), group_sizes)

    key, subkey = jrandom.split(key)
    u = jrandom.uniform(subkey, shape=(N,), minval=0, maxval=1)
    if random_T_star:
      key, subkey = jrandom.split(key)
      T_star_factors_ = T_star_factors(subkey, K)
    else:
      T_star_factors_ = jnp.array(T_star_factors)
    T_star_factors_per_item = jnp.repeat(T_star_factors_, group_sizes)
    T_star = -T_star_factors_per_item * jnp.log(u) / jnp.exp(X.dot(beta))

    key, subkey = jrandom.split(key)
    C = jrandom.exponential(subkey, shape=(N,)) * exp_scale
    delta = T_star <= C

    T = jnp.minimum(T_star, C)

    sorted_idx = jnp.argsort(-T)  # sort T descending

    T = jnp.take(T, sorted_idx, axis=0)
    X = jnp.take(X, sorted_idx, axis=0)
    delta = jnp.take(delta, sorted_idx, axis=0)
    group_labels = jnp.take(group_labels, sorted_idx, axis=0)

    # X = X - np.mean(X, axis=0)
    ret = (X, delta, beta, group_labels)
    if return_T:
      ret = (T,) + ret
    if return_T_star:
      T_star = jnp.take(T_star, sorted_idx, axis=0)
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
    group_sizes[K - N % K:] += 1

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


@functools.lru_cache(maxsize=None)
def full_data_generator(N: int,
                        K: int,
                        X_DIM: int,
                        T_star_factors: Optional[str] = None,
                        group_labels_generator_kind: str = "same",
                        group_X: str = "same",
                        exp_scale: float = 3.5):
  """Capable Cox data generation with string arguments.

  This function combines :py:func:`group_sizes_generator` and
  :py:func:`data_generator`.
  It also accepts ``T_star_factors`` etc. as strings, and performs necessary
  parsing with the :py:mod:`simpleeval` package.
  This function is ideally used directly by a command line interface.
  """

  group_sizes_gen = se.EvalWithCompoundTypes(
      functions={
          'custom':
              lambda *args: functools.partial(group_sizes_generator,
                                              group_labels_generator_kind=
                                              "manual",
                                              sizes=tuple(args))
      },
      names={
          'arithmetic_sequence':
              functools.partial(
                  group_sizes_generator,
                  group_labels_generator_kind="arithmetic_sequence",
                  start_val=N * 2 // (K * (K + 1))),
          'same':
              functools.partial(group_sizes_generator,
                                group_labels_generator_kind="same"),
      }).eval(group_labels_generator_kind)

  group_sizes = group_sizes_gen(N, K)

  X_generator = se.EvalWithCompoundTypes(
      functions={
          'normal':
              normal,
          'bernoulli':
              bernoulli,
          'custom':
              lambda gdists, dims, weights: functools.partial(
                  make_X_generator,
                  g_dists=gdists,
                  correlated_dims=dims,
                  correlated_weights=weights),
      },
      names={
          'group': grouping_X_generator_K3,
          'same': default_X_generator,
          'correlated': correlated_X_generator
      }).eval(group_X)

  if T_star_factors:
    T_star_factors = se.EvalWithCompoundTypes(
        functions={
            'fixed': lambda *args: tuple(args),
            'gamma': T_star_factors_inv_gamma_gen
        },
        names={
            'fixed': tuple((k + 1) / 2 for k in range(K)),
            'gamma': T_star_factors_inv_gamma_gen(1., 1.),
            'None': None
        }).eval(T_star_factors)

  if exp_scale == 'inf':
    exp_scale = onp.inf

  return group_sizes, data_generator(N,
                                     X_DIM,
                                     group_sizes,
                                     exp_scale=exp_scale,
                                     T_star_factors=T_star_factors,
                                     X_generator=X_generator,
                                     return_T=True,
                                     return_T_star=True)


def group_labels_to_indices(K, group_labels):
  """Computes group indices from ``group_labels``.

  Note: this function is done on CPU since JAX's current support on indexing.
  Therefore this WILL incurr extra host-device transfer
  if ``group_labels`` is on GPU.
  """
  group_labels = onp.array(group_labels)
  batch_mode = True
  if len(group_labels.shape) == 1:
    batch_mode = False
    group_labels = group_labels.reshape((1, -1))

  assert len(group_labels.shape) == 2
  #TODO make this more efficient?
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

  Args:
    group_labels: int array of shape ``(N,)``.
    data: a sequence of arrays, with a common prefix `(N, ...)` to be groupped.
    K: the number of groups.
    group_size: the maximum ``group_size``. If negative, the function will
      compute the quantity from ``group_labels``.

  Returns:
    a sequence of arrays, with the same structure as ``data``, where each array
    has a shape prefix of ``(K, group_size, ...)``.
  """
  if group_size < 0:
    group_size = jnp.max(jnp.vectorize(functools.partial(jnp.bincount,
                                                         length=K),
                                       signature="(N)->(K)")(group_labels),
                         axis=-1)

  return tuple(
      vutils.group_by_labels(group_labels, d, K=K, group_size=group_size)
      for d in data)


key = jrandom.PRNGKey(0)
key, data_generation_key = jrandom.split(key)


@functools.partial(jnp.vectorize, signature="(N,p),(p)->(N,p),(p),(p)")
def normalize(X, beta):
  """Cox model normalize.

  Normalizes ``X`` by subtracting mean; then scales ``X`` and ``beta``
  simultaneously by ``X``'s 1-norm.
  """
  X = X - jnp.mean(X, axis=0)
  scale = X.shape[0] / jnp.linalg.norm(X, ord=1, axis=0)
  X *= scale
  beta /= scale
  return X, beta, scale
