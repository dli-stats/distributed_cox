from oryx.core import reap, plant, sow


def remove_tag(fun, *, name: str):

  def wrapped(*args, **kwargs):
    intermediates = reap(fun, tag="tag1")(*args, **kwargs)
    print("=======")
    return plant(fun, tag="tag1")(intermediates, *args, **kwargs)

  return wrapped


def f(x):
  return sow(x, tag="tag2", name="b", mode='clobber')


print(reap(remove_tag(f, name="a"), tag="tag2")(1.))
