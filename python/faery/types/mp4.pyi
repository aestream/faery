import pathlib
import types
import typing

import numpy.typing

class Encoder:
    def __init__(
        self,
        path: typing.Union[pathlib.Path, str],
        dimensions: tuple[int, int],
        frame_rate: float,
        crf: float,
        preset: typing.Literal[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
            "placebo",
            "none",
        ],
        tune: typing.Literal[
            "film",
            "animation",
            "grain",
            "stillimage",
            "psnr",
            "ssim",
            "fastdecode",
            "zerolatency",
            "none",
        ],
        profile: typing.Literal[
            "baseline",
            "main",
            "high",
            "high10",
            "high422",
            "high444",
        ],
    ): ...
    def __enter__(self) -> Encoder: ...
    def __exit__(
        self,
        exception_type: typing.Optional[typing.Type[BaseException]],
        value: typing.Optional[BaseException],
        traceback: typing.Optional[types.TracebackType],
    ) -> bool: ...
    def write(
        self,
        frame: numpy.typing.NDArray[numpy.uint8],
    ): ...
