import click

from faery.stdio import StdEventInput, StdEventOutput


@click.command(name="output")
@click.argument("sink", type=str, default="stdout")
@click.pass_context
def cli_output(ctx, sink):
    if sink == "stdout":
        ctx.obj.output = StdEventOutput()

    if ctx.obj.input is None:
        ctx.obj.input = StdEventInput()
