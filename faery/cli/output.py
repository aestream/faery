import click

from faery.stdio import StdEventOutput
from faery.csv import CsvEventOutput


@click.command(name="output")
@click.argument("sink", type=str, default="stdout")
@click.pass_context
def cli_output(ctx, sink):
    """
    Sends data to varying output sinks. Supported sinks include:\n
    \b
      - stdout: Standard output
      - *.csv: A CSV file
    """
    if sink.endswith(".csv"):
        ctx.obj.output = CsvEventOutput(sink)
    elif sink == "stdout":
        ctx.obj.output = StdEventOutput()
    else:
        raise ValueError("Unsupported output sink", sink)
