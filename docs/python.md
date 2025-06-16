(usage-python)=
# Usage: Python

This page describes how to use Faery in a Python script to process event-based data.
Note that we assume you already installed Faery, see the [installation instructions](installation) if you haven't done so.

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

## Reading event data from input sources

Faery provides a set of input functions that lets you read event data from files, UDP streams, or other sources.

Here is a table with the available input functions:
| Input Function | Description |
|----------------|-------------|
| `faery.events_stream_from_array(...)` | Reads event data from a NumPy array |
| `faery.events_stream_from_camera(...)` | Reads event data from a camera (e.g., Prophesee, Intel) |
| `faery.events_stream_from_file(...)` | Reads event data from a file (e.g., AEDAT, RAW, DAT, ES, CSV) |
| `faery.events_stream_from_udp(...)` | Reads event data from a UDP stream |
| `faery.events_stream_from_stdin(...)` | Reads event data from standard input (stdin) |

## Working with event stream types

TODO

