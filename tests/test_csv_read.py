from io import StringIO

import faery


def test_read_csv():
    stream = faery.CsvFileEventStream("tests/data/test.csv")

    output = StringIO()
    stream.output("stdout", file=output)
    output.seek(0)
    assert output.read() == "1,1,1,0\n2,1,1,0\n3,1,1,1\n"
