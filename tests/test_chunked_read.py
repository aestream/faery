import faery
import numpy as np

def test_chunked_event_stream():
    stream = faery.read_file("tests/data/test.csv").chunk(n_events=2)

    chunk = next(iter(stream))

    assert len(chunk) == 2
    assert (chunk['t'] == np.array([[1], [3]])).all()
    assert (chunk['x'] == np.array([[10], [1]])).all()
    assert (chunk['y'] == np.array([[10], [1]])).all()
    assert (chunk['p'] == np.array([[False], [False]])).all()
