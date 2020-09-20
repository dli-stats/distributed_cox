import typer

app = typer.Typer()


@app.command()
def f(var_with_UPPERCASE: int, lowercase_var: int):
  return var_with_UPPERCASE + lowercase_var


app()
