import types
import typing

import numpy.typing

class Renderer:
    def __init__(
        self,
        dimensions: tuple[int, int],
        decay: typing.Literal["exponential", "linear", "window"],
        tau: int,
        colormap_type: typing.Literal["sequential", "diverging", "cyclic"],
        colormap_rgba: numpy.typing.NDArray[numpy.float64],
    ): ...
    def __enter__(self) -> Renderer: ...
    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool: ...
    def render(
        self, events: numpy.ndarray, render_t: int
    ) -> numpy.typing.NDArray[numpy.uint8]: ...
