import click

from faery.csv import CsvFileEventStream

from faery.stdio import StdEventInput


@click.command(name="input")
@click.argument("source", type=str, default="stdin")
@click.pass_obj
def cli_input(config, source):
    if source == "stdin":
        config.input = StdEventInput()
    elif source.endswith(".csv"):
        config.input = CsvFileEventStream(source)
    else:
        raise ValueError("Unsupported input source", source)
