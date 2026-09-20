import concurrent.futures

import numpy

import faery


def annotate(size: int):
    frame = numpy.zeros((64, 256, 4), dtype=numpy.uint8)
    frame[:, :, 3] = 255
    faery.image.annotate(frame, f"size {size}", 4, 4, size, (255, 255, 255, 255))
    assert frame[:, :, 0].any()


def test_annotate_from_many_threads():
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(annotate, (8 + index % 24 for index in range(2000))))
