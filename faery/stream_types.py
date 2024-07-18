import numpy

Event: numpy.dtype = numpy.dtype(
    [("t", "<u8"), ("x", "<u2"), ("y", "<u2"), (("p", "on"), "?")]
)

Events = numpy.ndarray

Frame = numpy.ndarray
