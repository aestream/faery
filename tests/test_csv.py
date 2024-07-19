from io import StringIO, IOBase
from tempfile import NamedTemporaryFile

import faery


def test_read_csv():
    stream = faery.read_file("tests/data/test.csv")

    output = StringIO()
    stream.output(faery.StdEventOutput(file=output))
    output.seek(0)
    assert output.read() == "1,10,10,0\n3,1,1,0\n10,20,20,1\n11,5,5,1\n"


def test_write_csv():
    stream = faery.read_file("tests/data/test.csv")
    with NamedTemporaryFile(mode="w+") as fp:
        stream.output(faery.CsvEventOutput(fp.file))
        fp.seek(0)
        assert fp.read() == "1,10,10,0\n3,1,1,0\n10,20,20,1\n11,5,5,1\n"
