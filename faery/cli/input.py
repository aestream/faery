import click

from faery.inputs import read_file

from faery.stdio import StdEventInput


@click.command(name="input")
@click.argument("source", type=str, default="stdin")
@click.pass_obj
def cli_input(config, source):
    if source == "stdin":
        config.input = StdEventInput()
    elif source.endswith(".csv") or source.endswith(".dat"):
        config.input = read_file(source)
    else:
        raise ValueError("Unsupported input source", source)
