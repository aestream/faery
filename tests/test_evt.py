from faery import read_file


def test_evt_read():
    first = []
    last = []
    for ev in read_file("tests/data/sample.raw"):
        if len(first) == 0:
            first = ev
        else:
            last = ev
    print(first)
    assert list(first[0]) == [14882464, 891, 415, 1]
    assert list(last[-1]) == [15582463, 927, 586, 0]
