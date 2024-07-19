from .output import EventOutput
from .types import Events


class StdEventOutput(EventOutput):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def apply(self, data: Events):
        for event in data:
            print(f"{event['t']},{event['x']},{event['y']},{int(event['p'])}", **self.kwargs)
        print("", **self.kwargs)
