from jax import custom_vjp, custom_jvp, jacfwd, jacrev


def f(x):
  return x


f = custom_jvp(f)
f.defjvp(lambda *args: args)

f = custom_vjp(f)


def fwd(*primals):
  return (primals, 0.)


def bwd(res, ct):
  return ct


f.defvjp(fwd, bwd)

print(jacfwd(f)(1.))
print(jacrev(f)(1.))