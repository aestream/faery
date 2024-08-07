from tempfile import NamedTemporaryFile

import numpy

from faery import read_file, output_file
from faery import rusty_faery as rusty


def test_dat_read():
    first = []
    last = []
    for ev in read_file("tests/data/sample.dat"):
        if len(first) == 0:
            first = ev
        else:
            last = ev
    assert list(first[0]) == [0, 237, 121, 1]
    assert list(last[-1]) == [50000, 210, 142, 1]


def test_dat_write():
    stream = read_file("tests/data/sample.dat")
    with NamedTemporaryFile(mode="w+", suffix=".dat", ) as fp:
        print(fp.name)
        output = output_file(fp.name, dimensions=(640, 480))
        stream.output(output)

        for ev in read_file(fp.name):
            assert list(ev[0]) == [0, 237, 121, 1]
            break
