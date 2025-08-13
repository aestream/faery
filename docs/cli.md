(usage-cli)=
# Usage: Command line

Faery provides a command line interface (CLI) for converting and processing neuromorphic event-based data.
It's particularly useful for quick conversions, rendering videos, and batch processing of multiple files.
This page gives you an overview of the available CLI commands.
We assume you already installed Faery, see the [installation instructions](#installation) if you haven't done so.

## Usage examples
Here are some examples of how to use Faery from the command line:

Convert a Prophesee raw file (input.raw) to AEDAT (output.aedat4)
```sh
faery input file input.raw output file output.aedat4
```

Render an event file (input.es) as a real-time video (output.mp4)
```sh
faery input file input.es \
      filter regularize 60.0 \
      filter render exponential 0.2 starry_night \
      output file output.mp4
```

Render an event file (input.es) as a video 10 x slower than real-time (output.mp4)
The second render parameter (0.03) is the exponential decay constant.
Slow-motion videos look better with shorter decays but it does not need to be scaled like regularize, which controls the playback speed.
```sh
faery input file input.es \
      filter regularize 600.0 \
      filter render exponential 0.03 starry_night \
      output file output.mp4
```

Render an event file (`input.es`) as frames (`frames/*.png`)
```sh
faery input file input.es \
      filter regularize 60.0 \
      filter render exponential 0.2 starry_night \
      output files 'frames/{index:04}.png'
```

Print ON events to the terminal
```sh
faery input file input.aedat4 filter remove-off-events
```

Read event data from UDP and write it to a CSV file (output.csv)
```sh
faery input udp 0.0.0.0:3000 output file output.csv
```
