from io import StringIO

import faery

def test_read_csv():
  stream = faery.CsvFileEventStream("tests/data/test.csv")

  output = StringIO()
  stream.output("stdout", file=output)
  output.seek(0)
  assert output.read() == "1,10,10,0\n3,1,1,0\n10,20,20,1\n11,5,5,1\n"