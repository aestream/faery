from .stream import Stream, StreamIterator
from .types import Frame


class FrameStream(Stream[Frame]):
    pass


class FrameStreamIterator(StreamIterator[Frame]):
    pass
