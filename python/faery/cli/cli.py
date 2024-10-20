import click

from faery.cli import CliConfig

# from faery.cli.input import cli_input
# from faery.cli.output import cli_output

# from faery.stdio import StdEventOutput


@click.group(chain=True)
@click.pass_context
def cli(ctx):
    """\b
         __       __      _____
        /  \     /  \    |  ___|_ _  ___ _ __ _   _
        | ( \___/ ) |    | |_ / _` |/ _ \ '__| | | |
         \__/   \__/     |  _| (_| |  __/ |  | |_| |
           _\___/_       |_|  \__,_|\___|_|   \__, |
          (_/   \_)                           |___/

    Faery is an "event data switchboard" that converts Neuromorphic camera data (events) between formats.
    It can also generate videos, spectrograms, and event rate curves.
    \n"""
    ctx.ensure_object(CliConfig)


# @cli.result_callback()
# @click.pass_obj
# def process(config, processors):
#     if config.input is None:
#         raise ValueError("No input specified")

#     if config.output is None:
#         config.output = StdEventOutput()

#     for data in config.input:
#         config.output.apply(data)


# input_stream = cli.add_command(cli_input)
# output_type = cli.add_command(cli_output)
