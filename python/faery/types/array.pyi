import typing

import numpy
import numpy.typing

def concatenate(
    parts: typing.Iterable[numpy.ndarray], dtype: numpy.typing.DTypeLike
) -> typing.Optional[numpy.ndarray]: ...
