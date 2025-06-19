(usage-python)=
# Usage: Python

This page describes how to use Faery in a Python script to process event-based data.
Note that we assume you already installed Faery, see the [installation instructions](#installation) if you haven't done so.

The Python code generally follows the same structure as the command line interface (CLI), where you have some input, some filtering/manipulation, and then an output.

```python
import faery

input_stream = faery.SOME_INPUT(...)
filtered_stream = input_stream.SOME_FILTER(...)
filtered_stream.SOME_OUTPUT(...)
```

Which can be simplified to:

```python
import faery
faery.SOME_INPUT(...) \
     .SOME_FILTER(...) \
     .SOME_OUTPUT(...)
```

Of course, there are many variations to this, and you can chain multiple filters and outputs together.
Below, we will go through some common use cases and examples of how to use Faery in a Python script.

## Inputs: How to read data

Faery provides a set of input functions that lets you read event data from files, UDP streams, or other sources.

Here is a table with the available input functions:
| Input Function | Description |
|----------------|-------------|
| `faery.events_stream_from_array(...)` | Reads event data from a NumPy array |
| `faery.events_stream_from_camera(...)` | Reads event data from a camera (e.g., Prophesee, Intel) |
| `faery.events_stream_from_file(...)` | Reads event data from a file (e.g., AEDAT, RAW, DAT, ES, CSV) |
| `faery.events_stream_from_udp(...)` | Reads event data from a UDP stream |
| `faery.events_stream_from_stdin(...)` | Reads event data from standard input (stdin) |

## Outputs: How to write data

Faery provides a set of output functions that lets you write event data to files, UDP streams, or other sources.

Here is a table with the available output functions:
| Output Function | Description |
|-----------------|-------------|
| `stream.to_array(...)` | Writes event data to a NumPy array |
| `stream.to_file(...)` | Writes event data to a file (e.g., AEDAT, RAW, DAT, ES, CSV) |
| `stream.to_udp(...)` | Writes event data to a UDP stream |
| `stream.to_stdout(...)` | Writes event data to standard output (stdout) |

## Filters: How to process data

Faery provides a set of filter functions that lets you process event data in various ways.
Note that the filters depend on the *type* of stream you are operating on, so we recommend you familiarize yourself with the different types of streams and their respective filters: @stream-types.

Here are a few examples of filter functions for events:

| Filter function | Supported stream types | Description |
|-----------------|-------------|-------------|
| `stream.crop(...)` | EventStream, FiniteEventStream, RegularEventStream, FiniteRegularEventStream | Removes events outside a given spatial range |
| `stream.chunks(...)` | EventStream, FiniteEventStream, RegularEventStream, FiniteRegularEventStream | Splits the stream into parts of a given number of events |
| `stream.map(...)` | EventStream, FiniteEventStream, RegularEventStream, FiniteRegularEventStream | Applies a function to each package in the stream |
| `stream.time_slice(...)` | EventStream, FiniteEventStream | Extracts a time slice from the stream |

And here are a few examples of filter functions for images:

| Filter function | Supported stream types | Description |
|-----------------|-------------|-------------|
| `stream.scale(...)` | FrameStream, FiniteFrameStream, RegularFrameStream, FiniteRegularFrameStream | Scales the image to a given size |
| `stream.annotate(...)` | FrameStream, FiniteFrameStream, RegularFrameStream, FiniteRegularFrameStream | Adds text annotations to the images |
| `stream.add_timecode(...)` | FrameStream, FiniteFrameStream, RegularFrameStream, FiniteRegularFrameStream | Adds a textual timecode to the images |
| `stream.add_overlay(...)` | FrameStream, FiniteFrameStream, RegularFrameStream, FiniteRegularFrameStream | Adds a visual overlay to the images |

## Putting it all together

Once you get a feel for the inputs, filters, and outputs, working with Faery is easy!
You can mix and match any filters and outputs as you want - if you make sure the type of the stream matches the type of operation you are doing.
For instance, it doesn't make sense to use `stream.chunks(...)` on a stream of frames.
Luckily, Faery will let you know if you are using an unsupported operation on a stream of frames, so feel free to experiment!

Here is an example where we stream events from a file, crop them, and save the result to a CSV file:

```python
(
    faery.events_stream_from_file(
        "dvs.es",
    )
    .crop(
        left=110,
        right=210,
        top=70,
        bottom=170,
    )
    .time_slice(
        start="00:00:00.400000",
        end="00:00:00.600000",
        zero=True,
    )
    .to_file(
        "dvs_crop_and_slice.csv",
    )
)
```

Here is another example where we stream events from an event camera to a UDP socket without using an intermediate filter (note that this requires the [`event-camera-drivers` package](https://aestream.github.io/faery/install):

```python
(
    faery.events_stream_from_camera("Inivation")  # Open an Inivation camera
    .to_udp(
        host="localhost",
        port=5005,
    )
)
```

And here is an example where we stream events from a file, regularize them, render them to frames, and save the result to an MP4 video file:

```python
(
    faery.events_stream_from_file(
        "dvs.es",
    )
    .regularize(frequency_hz=60.0)
    .render(
        decay="exponential",
        tau="00:00:00.200000",
        colormap=faery.colormaps.managua.flipped(),
    ) # This is now a stream of frames!
    .to_file("dvs.mp4")
)
```
