import click

from faery.events_input import read_file

from faery.stdio import StdEventInput


@click.command(name="input")
@click.argument("source", type=str, default="stdin")
@click.pass_obj
def cli_input(config, source):
    """
    Reads from varying input sources. Supported sources include:\n
    \b
      - stdin: Standard input
      - *.csv: A CSV file
      - *.dat: A DAT file
      - *.es: An ES file
      - *.raw/*.evt: An EVT file
    """
    if source == "stdin":
        config.input = StdEventInput()
    elif (
        source.endswith(".csv")
        or source.endswith(".dat")
        or source.endswith(".es")
        or source.endswith(".raw")
        or source.endswith(".evt")
    ):
        config.input = read_file(source)
    else:
        raise ValueError("Unsupported input source", source)