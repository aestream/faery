from faery.stream import Stream, StreamIterator
from faery.stream_types import Frame


class FrameStream(Stream[Frame]):
    pass


class FrameStreamIterator(StreamIterator[Frame]):
    pass
