from typing import Optional, Union

from faery.output import EventOutput
from faery.stream import Stream, StreamIterator
from faery.stream_types import Events


# def _output_from_str(output: str, **kwargs) -> Optional[EventOutput]:
#     if output == "stdout":
#         return StdEventOutput(**kwargs)
#     return None


class EventStream(Stream[Events]):

    def output(self, output: EventOutput, **kwargs) -> None:
        # if isinstance(output, str):
            # output = _output_from_str(output, **kwargs)

        if not isinstance(output, EventOutput):
            raise ValueError(f"Unknown output: {output}")

        for data in self:
            output.apply(data)


class EventStreamIterator(StreamIterator[Events]):
    pass


class ChunkedEventStream(Stream[Events]):
    pass


class ChunkedEventStreamIterator(StreamIterator[Events]):
    pass
