![faery logo](faery_logo.png)

# Faery: A Stream Processing Library for Neuromorphic Event-Based Data

[![PyPI - Downloads](https://img.shields.io/pypi/dm/faery?logo=pypi)](https://pypi.org/project/faery/)
[![GitHub Tag](https://img.shields.io/github/v/tag/aestream/faery?logo=github)](https://github.com/aestream/faery/releases)
[![Discord](https://img.shields.io/discord/1044548629622439977)](https://discord.gg/C9bzWgNmqk)

Faery converts neuromorphic event-based data between formats. It can also generate videos, spectrograms, and event rate curves.

📄 [Read more in our documentation](https://aestream.github.io/faery).

## Installation

Via pip: `pip install faery`.

We recommend using a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to install Faery.
More information in the [installation instructions](https://aestream.github.io/faery/install).

## Usage

Faery can be used as a [**command line tool**](https://aestream.github.io/faery/cli) or as a [**Python library**](https://aestream.github.io/faery/python).

As a command line tool, Faery can convert data between formats, render videos, and analyze event-based data. It is particularly useful for quick conversions and batch processing of multiple files.
As an example, you can convert a Prophesee raw file to AEDAT format with:

```sh
faery input file input.raw output file output.aedat4
```

As a Python library, Faery provides a set of input functions to read event data from files, UDP streams, or other sources. You can chain methods to filter, render, or analyze the data. For example, to render an AEDAT4 event file as a real-time MP4 video:

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
