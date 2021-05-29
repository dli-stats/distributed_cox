"""Simple CMD."""

from types import FunctionType

import inspect
import argparse


def _default_handle(typ):

  def handle_type(_):
    return dict(type=typ)

  return typ, handle_type


def _handle_bool(param: inspect.Parameter):
  if param.default != inspect.Parameter.empty:
    if param.default is True:
      action = "store_false"
    elif param.default is False:
      action = "store_true"
    else:
      raise ValueError("Invalid boolean default value @ {}".format(param))
    return dict(action=action)
  return dict(type=bool)


class SimpleCMD:
  """Simple CommandLine Creation through type annotations."""

  DEFAULT_ARG_HANDLERS = [
      _default_handle(int),
      _default_handle(float),
      _default_handle(str), (bool, _handle_bool),
      (lambda param: True, lambda param: dict(type=param.annotation))
  ]

  def __init__(self):
    self.parser = argparse.ArgumentParser()
    self.subparsers = self.parser.add_subparsers()
    self._arg_handlers = list(self.DEFAULT_ARG_HANDLERS)

  def arg_handler(self, matcher):
    """Defines an argument handler as a decorator."""

    def wrapped(fun):
      self._arg_handlers.append((matcher, fun))

    return wrapped

  def command(self, fun):
    """Turns fun into a sub-command."""
    cmd_parser = self.subparsers.add_parser(fun.__name__)
    fun_sig = inspect.signature(fun, follow_wrapped=True)
    for param in fun_sig.parameters.values():
      self._handle_param(cmd_parser, param)

    def call_fun(args):
      pos_args = []
      kw_args = {}
      for param in fun_sig.parameters.values():
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                          inspect.Parameter.POSITIONAL_OR_KEYWORD):
          pos_args.append(getattr(args, param.name))
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
          kw_args[param.name] = getattr(args, param.name)
        else:
          raise ValueError()
      return fun(*pos_args, **kw_args)

    cmd_parser.set_defaults(func=call_fun)

  def _handle_param(self, parser: argparse.ArgumentParser,
                    param: inspect.Parameter):
    """Addes argument to parser for param."""
    if param.annotation is None:
      raise ValueError("{} needs an annotation".format(param))
    for matcher, handler in self._arg_handlers:
      matched = ((param.annotation == matcher) or
                 (isinstance(matcher, FunctionType) and matcher(param)))
      if matched:
        kwargs = handler(param)
        if param.default == inspect.Parameter.empty:
          names = [param.name]
          default = None
        else:
          names = ["--{}".format(param.name)]
          default = param.default
        return parser.add_argument(*names, default=default, **kwargs)
    raise ValueError("No handler for {}".format(param))

  def run(self):
    """Runs the command line application."""
    args = self.parser.parse_args()
    return args.func(args)


if __name__ == "__main__":

  cmd = SimpleCMD()

  @cmd.command
  def do_f(x: int, y: int, test_bool: bool = False):
    if test_bool:
      return print(x + y)
    else:
      return print(x * y)

  cmd.run()
