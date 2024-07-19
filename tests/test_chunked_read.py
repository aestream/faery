import faery
from io import StringIO


def test_chunked_event_stream():
    stream = faery.CsvFileEventStream("tests/data/test.csv").chunk(n_events=2)

    output = StringIO()
    stream.output("stdout", file=output)
    output.seek(0)
    assert output.read() == "1,10,10,0\n3,1,1,0\n\n10,20,20,1\n11,5,5,1\n\n"