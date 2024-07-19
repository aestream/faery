from faery import read_file


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