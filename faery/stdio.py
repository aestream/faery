from faery.output import EventOutput
from faery.stream_types import Events
from faery.stream_event import EventStream, EventStreamIterator


class StdEventOutput(EventOutput):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, data: Events):
        for event in data:
            print(
                f"{event['t']},{event['x']},{event['y']},{int(event['p'])}",
                **self.kwargs,
            )


class StdEventInput(EventStream):

    def __iter__(self):
        return StdEventInputIterator()


class StdEventInputIterator(EventStreamIterator):

    def __next__(self):
        line = input()
        if not line:
            raise StopIteration
        t, x, y, p = line.split(",")
        return {"t": t, "x": x, "y": y, "p": bool(int(p))}
