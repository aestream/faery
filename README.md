![faery logo](faery_logo.png)

# Faery: A Stream Processing Library for Neuromorphic Event-Based Data

[![PyPI - Downloads](https://img.shields.io/pypi/dm/faery?logo=pypi)](https://pypi.org/project/faery/)
[![GitHub Tag](https://img.shields.io/github/v/tag/aestream/faery?logo=github)](https://github.com/aestream/faery/releases)
[![Discord](https://img.shields.io/discord/1044548629622439977)](https://discord.gg/C9bzWgNmqk)
[![Neuromorphic Computing](https://img.shields.io/badge/Collaboration_Network-Open_Neuromorphic-blue)](https://open-neuromorphic.org/neuromorphic-computing/)


Faery converts neuromorphic event-based data between formats. It can also generate videos, spectrograms, and event rate curves.

📄 [Read more in our documentation](https://aestream.github.io/faery).

## Installation

Using `pip`: `pip install faery`.

We recommend using a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to install Faery.
More information in the [installation instructions](https://aestream.github.io/faery/install).

## Usage

Faery can be used as a [**command line tool**](https://aestream.github.io/faery/cli) or as a [**Python library**](https://aestream.github.io/faery/python).

**Command line tool**: Faery can convert data between formats, render videos, and analyze event-based data. It is particularly useful for quick conversions and batch processing of multiple files.
Here are three examples:

1. Convert a Prophesee raw file to AEDAT format:

```sh
faery input file input.raw output file output.aedat4
```

2. Create an MP4 video from your event data with temporal filtering:

```sh
faery input file events.aedat4 filter temporal --window-size 1000us output mp4 output.mp4 --frame-rate 30
```

3. Stream data from an [Inivation camera](https://inivation.com/) to a UDP socket (note: [requires event camera drivers](https://aestream.github.io/faery/install)):

```sh
faery input inivation camera output udp localhost 7777
```

**Python library**: Faery provides a set of input functions to read event data from files, UDP streams, or other sources. You can chain methods to filter, render, or analyze the data. For example, to render an AEDAT4 event file as a real-time MP4 video:

```python
import faery
faery.events_stream_from_file("input.aedat4") \
    .regularize(frequency_hz=60.0) \
    .render(exponential_decay=0.2, style="starry_night") \
    .to_mp4("output.mp4")
```

More information is available in the [command line usage documentation](https://aestream.github.io/faery/cli), the [Python library documentation](https://aestream.github.io/faery/python), and the [examples directory](https://github.com/aestream/faery/tree/main/examples).

## Acknowledgements

Faery was initiated at the [2024 Telluride neuromorphic workshop](https://sites.google.com/view/telluride-2024/) by

-   [Alexandre Marcireau](https://github.com/amarcireau)
-   [Jens Egholm Pedersen](https://github.com/jegp)
-   [Gregor Lenz](https://github.com/biphasic)
-   [Gregory Cohen](https://github.com/gcohen)

License: LGPLv3.0
