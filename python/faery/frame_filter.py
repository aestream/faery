import collections.abc
import types
import typing

import numpy

from . import frame_stream
from . import stream


def restrict(
    prefixes: set[
        typing.Literal[
            "Float64",
            "FiniteFloat64",
            "RegularFloat64",
            "FiniteRegularFloat64",
            "Rgba8888",
            "FiniteRgba8888",
            "RegularRgba8888",
            "FiniteRegularRgba8888",
        ]
    ]
):
    def decorator(method):
        method._prefixes = prefixes
        return method

    return decorator


FILTERS: dict[str, typing.Any] = {}


def typed_filter(
    prefixes: set[
        typing.Literal[
            "Float64",
            "FiniteFloat64",
            "RegularFloat64",
            "FiniteRegularFloat64",
            "Rgba8888",
            "FiniteRgba8888",
            "RegularRgba8888",
            "FiniteRegularRgba8888",
        ]
    ]
):
    def decorator(filter_class):
        attributes = [
            name
            for name, item in filter_class.__dict__.items()
            if isinstance(item, types.FunctionType)
        ]
        for prefix in prefixes:

            class Generated(getattr(frame_stream, f"{prefix}FrameFilter")):
                pass

            for attribute in attributes:
                method = getattr(filter_class, attribute)
                if not hasattr(method, "_prefixes") or prefix in method._prefixes:
                    setattr(Generated, attribute, getattr(filter_class, attribute))
            Generated.__name__ = f"{prefix}{filter_class.__name__}"
            Generated.__qualname__ = Generated.__name__
            FILTERS[Generated.__name__] = Generated
        return None

    return decorator
