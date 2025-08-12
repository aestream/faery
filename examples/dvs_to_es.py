import argparse
import pathlib
import numpy

import faery

parser = argparse.ArgumentParser()
parser.add_argument("input")
args = parser.parse_args()

input = pathlib.Path(args.input)
output = input.parent / f"{input.stem}.es"

events = numpy.fromfile(
    input,
    dtype=[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("on", "?")],
)

events["t"] -= events["t"][0]

with faery.event_stream.Encoder(
    path=output, event_type="dvs", zero_t0=False, dimensions=(1280, 720)
) as encoder:
    encoder.write(events)
