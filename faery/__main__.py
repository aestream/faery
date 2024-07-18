import click

from faery.cli import CliConfig
from faery.cli.input import cli_input
from faery.cli.output import cli_output

from faery.stdio import StdEventOutput


@click.group(chain=True)
@click.pass_context
def cli(ctx):
    """
    Faery is a library that ferries event data and frames from A to B.
    By default, we stream from STDIN to STDOUT without modifying the data,
    but the behavior can be customized using the following pattern:

    \b
        faery [input ...] [filter ...] [output ...]

    \b
                           __/___                 _____
                     _____/______|               |  ___|_ _  ___ _ __ _   _
     _______    _____/_____\_______\_____        | |_ / _` |/ _ \ '__| | | |
     \      \  |               |      \   \      |  _| (_| |  __/ |  | |_| |
    ~~~~~~~~~~~~~~ ~~~~~ ~~~ ~~~~~~~~ ~~~~~ ~~~  |_|  \__,_|\___|_|   \__, |
         ~~~~  ~~~~~   ~~~   ~~  ~~~~~  ~~~ ~~                        |___/
    \n"""
    ctx.ensure_object(CliConfig)


@cli.result_callback()
@click.pass_obj
def process(config, processors):
    if config.input is None:
        raise ValueError("No input specified")

    if config.output is None:
        config.output = StdEventOutput()

    for data in config.input:
        config.output.apply(data)


def main():
    input_stream = cli.add_command(cli_input)
    output_type = cli.add_command(cli_output)
    c = cli()


if __name__ == "__main__":
    main()
