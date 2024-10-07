import click

from faery.stdio import StdEventOutput
from faery.csv import CsvEventOutput
from faery.udp_encoder import UdpEventOutput


@click.command(name="output")
@click.argument("sink", type=str, default="stdout")
@click.argument("address", type=str, default=None)
@click.argument("port", type=int, default=None)
@click.pass_context
def cli_output(ctx, address, port, sink):
    """
    Sends data to varying output sinks. Supported sinks include:\n
    \b
      - stdout: Standard output
      - udp [server] [port]: UDP packets with SPIF encoding
      - *.csv: A CSV file
    """
    if sink.endswith(".csv"):
        ctx.obj.output = CsvEventOutput(sink)
    elif sink == "stdout":
        ctx.obj.output = StdEventOutput()
    elif sink == "udp":
        if address is None or port is None:
            raise ValueError("UDP output requires address and port arguments")
        ctx.obj.output = UdpEventOutput(server_address=address, server_port=port)
    else:
        raise ValueError("Unsupported output sink", sink)
