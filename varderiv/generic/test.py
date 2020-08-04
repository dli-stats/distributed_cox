from jax import make_jaxpr

import jax.numpy as np


def f(x, y):
  m = np.sin(y)
  return np.sin(x)


print(make_jaxpr(f)(np.zeros(1), np.zeros(1)))