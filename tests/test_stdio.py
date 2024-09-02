import signal
import subprocess
import sys
from io import StringIO
import tempfile

import numpy as np

import pytest

from faery.stream_event import StreamIterator
from faery.stdio import StdEventOutput, StdEventInput
from faery.csv import CsvEventStream


def test_StdEventOutput_apply():
    output = StdEventOutput()
    events = next(iter(CsvEventStream("tests/data/sample.csv")))
    output.apply(events)
    captured_output = StringIO()
    sys.stdout = captured_output
    output.apply(events)
    sys.stdout = sys.__stdout__
    stdout_value = captured_output.getvalue()
    with open("tests/data/sample.csv") as f:
        expected_output = f.read()
    assert stdout_value == expected_output


def test_StdEventInputIterator_next():
    with open("tests/data/sample.csv") as f:
        event_string = f.read()
    event_io = StringIO(str(event_string))
    event_expected = next(iter(CsvEventStream("tests/data/sample.csv")))
    StreamIterator.BUFFER_SIZE = len(event_expected)
    input_stream = StdEventInput(file=event_io)
    input_iterator = iter(input_stream)
    actual = next(input_iterator)
    assert np.array_equal(actual, event_expected)
    assert len(input_iterator.buffer) == 0

    with pytest.raises(StopIteration):
        next(input_iterator)


def test_StdEventInput_terminates():
    with tempfile.NamedTemporaryFile() as f:
        process = subprocess.Popen(
            ["python", "tests/test_stdio.py", str(f.name)],
            stdin=subprocess.PIPE,
            text=True,
        )

        # Send event data
        stdout, stderr = process.communicate("1,2,3,1\n4,5,6,0\n7,8,9,1\n")
        assert stdout is None
        assert stderr is None

        process.wait()  # Test that it terminates


if __name__ == "__main__":
    # Used in test_StdEventInput_terminates
    with open(sys.argv[1]) as f:
        event_string = "1,2,3\n4,5,6\n7,8,9\n"
        event_iterator = next(iter(StdEventInput(file=f)))
