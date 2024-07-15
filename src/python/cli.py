from stream import Sink, Events


class StdOutSink(Sink):

    def process(self, data: Events) -> None:
        for event in data:
            print(event)