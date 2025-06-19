(stream-types)=
# Stream types in Faery

Faery supports streams of various data types with different characteristics.
Understanding these streams is crucial for understanding the behavior of Faery---and in particular the Python API.

## Stream types

All streams are implemented as a [`Stream`](https://github.com/aestream/faery/blob/main/python/faery/stream.py#L9) which iterates over *some* data.
The data type characterizes what is contained in the stream.
We mostly operate with two types: events and frames:

| Stream | Data Type | Characteristics |
|--------|-----------|-----------------|
| EventsStream | `np.ndarray` | Sparse event data represented as [timestamps, 2-d coordinates, and polarity bit](https://github.com/aestream/faery/blob/main/python/faery/events_stream.py#L16) `(t, x, y, p)`. |
| FrameStream | [`Frame`](https://github.com/aestream/faery/blob/main/python/faery/frame_stream.py#L19) | Dense frame data represented as `[timestamp, np.ndarray`](https://github.com/aestream/faery/blob/main/python/faery/frame_stream.py#L165). |


## Finite, infinite, and regular streams
Apart from the type of data, streams can have different characteristics that matter a great deal for what you can do with them.
For instance, a finite stream can be processed in a single pass, while an infinite stream requires some kind of continuous processing.
Additionally, streams can be structured in time by sending data at regular intervals.

| Stream type | Description | Examples |
|-------------|-------------|-----------------|
| InfiniteStream | An infinite stream requires some kind of continuous processing. | Reading from a camera or UDP source. |
| FiniteStream | A finite stream can be processed in a single pass. | Reading from a file or a finite source. |
| RegularStream | A regular stream sends data at regular intervals. | Filtering an event stream to output events at regular intervals. |
| FiniteRegularStream | A finite regular stream sends data at regular intervals and can be processed in a single pass. | Filtering a finite event stream to output events at regular intervals. |
