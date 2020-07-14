"""Generic distributed."""

import jax.numpy as np
from jax.experimental.jet import jet


def test_model(X, Y, beta):
  return np.sum((Y - X.dot(beta))**2)


def group_distributed(f, primals, series):
  pass
