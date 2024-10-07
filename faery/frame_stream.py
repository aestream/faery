import collections.abc
import typing

import numpy

from . import stream

StreamType = typing.TypeVar("StreamType")


class GenericFrameStream(typing.Generic[StreamType]):
    def __iter__(self) -> collections.abc.Iterator[numpy.ndarray]:
        raise NotImplementedError()

    def dimensions(self) -> tuple[int, int]:
        raise NotImplementedError()

    def crop(self, left: int, right: int, top: int, bottom: int) -> StreamType:
        raise Exception("not implemented yet")  # @TODO

    def mask(self, array: numpy.ndarray) -> StreamType:
        raise Exception("not implemented yet")  # @TODO

    def transpose(
        self,
        action: typing.Literal[
            "flip_left_right",
            "flip_bottom_top",
            "rotate_90_counterclockwise",
            "rotate_180",
            "rotate_270_counterclockwise",
            "flip_up_diagonal",
            "flip_down_diagonal",
        ],
    ) -> StreamType:
        raise Exception("not implemented yet")  # @TODO

class FrameStream(
    stream.UniformStream[numpy.ndarray],
    GenericFrameStream["FrameStream"],
):
    pass

class FiniteFrameStream(
    stream.FiniteUniformStream[numpy.ndarray],
    GenericFrameStream["FrameStream"],
):
    pass
