# Quick start

Faery is a swiss army knife for neuromorphic event-based data. It can convert data between formats, stream event camera data, generate videos, and visualize spectrograms and event rate curves.

It is accessible in two forms: as a command line tool and as a Python library. This quick start guide will help you get started with both.
We assume you already installed Faery, see the [installation instructions](@installation) if you haven't done so.

## Using Faery from the command line

After installing Faery, you should have access to the `faery` command in your terminal.
This gives you access to several commands to convert, render, and analyze event-based data.

We'll start with two modes for using Faery to convert and process data, and refer to the [command line usage documentation](#usage-cli) for more details.

### Converting data via the command line

To convert data between formats, you can use the `faery input ... output` command structure.
For example, to convert a Prophesee raw file to AEDAT format, you can run:

```sh
faery input file input.raw output file output.aedat4
```

You can add filters to the command to process the data further. For example, to render an event file as a real-time video, you can run:

```sh
faery input file input.es filter regularize 60.0 filter render exponential 0.2 starry_night output file output.mp4
```

### Processing files via the command line

Faery also has a powerful batch-processing mode that allows you to process many files at once.
This is useful for working with many files at once or for automating repetitive tasks.

To use this mode, run `faery init` in a directory containing your recordings.
Faery will generate a script that you can edit to make sure you're getting the expected result and, when you're ready, run with `faery run`.

## Using Faery in a Python script

You can also use Faery as a Python library to process event-based data in your scripts.
To get started, you can import Faery in your Python script:

```python
import faery
```

Faery provides a set of input functions that lets you read event data from files, UDP streams, or other sources.
For example, to read an event file and render it as a video, you can do:
```python
import faery

faery.events_stream_from_file("input.es")
```

Depending on what you want to do, you can chain methods to filter, render, or analyze the data.
For example, to render an AEDAT4 event file as a real-time MP4 video, you can do:
```python
import faery
faery.events_stream_from_file("input.aedat4") \
    .regularize(frequency_hz=60.0) \
    .render(decay="exponential",
           tau="00:00:00.002000",
           colormap=faery.colormaps.starry_night) \
    .to_file("output.mp4")
```
:::{figure} mp4_example.mp4
An example of a generated `.mp4` file from a an event-based input stream.
:::

Or, if you want to generate a sequence of PNG images from an event file, you can do:

```python
(
    faery.events_stream_from_file(
        faery.dirname.parent / "tests" / "data" / "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .render(
        decay="exponential",
        tau="00:00:00.200000",
        colormap=faery.colormaps.managua.flipped(),
    )
    .to_files(
        faery.dirname.parent
        / "tests"
        / "data_generated"
        / "dvs_frames"
        / "{index:04}.png",
    )
)
```

:::{figure} frame_example.png
One of the generated frames from the example above.
Note the exponential decay of events over time.
:::

Or, if you want to generate the event rate curve and save it as a PNG image, you can do:
```python
import faery
faery.events_stream_from_file("input.es") \
    .to_event_rate() \
    .to_file("output.png")
```

You can find more examples in @usage-python and the [examples directory](https://github.com/aestream/faery/tree/main/examples).
