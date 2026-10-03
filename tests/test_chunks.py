from __future__ import annotations

import itertools

import numpy
import pytest

import faery

DIMENSIONS = (10, 8)


def _events(count: int) -> numpy.ndarray:
    events = numpy.zeros(count, dtype=faery.EVENTS_DTYPE)
    events["t"] = numpy.arange(count) * 10
    events["x"] = numpy.arange(count) % DIMENSIONS[0]
    events["y"] = numpy.arange(count) % DIMENSIONS[1]
    events["on"] = numpy.arange(count) % 2 == 0
    return events


def _chunks(stream, chunk_length: int) -> list[numpy.ndarray]:
    # A bounded read: chunks once looped forever, yielding empty slices,
    # whenever a packet was at least chunk_length long. Reading at most
    # `limit` packets turns that hang into a failure.
    limit = 1000
    packets = list(itertools.islice(stream.chunks(chunk_length), limit))
    assert len(packets) < limit, "chunks yields an endless stream of packets"
    return packets


@pytest.mark.parametrize(
    "packet_lengths, chunk_length",
    [
        ([10], 3),  # one packet longer than a chunk
        ([6], 6),  # a packet exactly one chunk long
        ([12], 4),  # a packet of exactly several chunks
        ([2, 7, 1, 5], 3),  # packets shorter and longer than a chunk
        ([5, 5, 5], 1),  # one-event chunks
        ([0, 4, 0, 4], 3),  # empty packets in between
    ],
)
def test_chunks_repartitions_into_fixed_lengths(packet_lengths, chunk_length):
    events = _events(sum(packet_lengths))
    boundaries = numpy.cumsum([0] + packet_lengths)
    packets = [events[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]

    class Packets(faery.FiniteEventsStream):
        def __iter__(self):
            yield from (packet.copy() for packet in packets)

        def dimensions(self):
            return DIMENSIONS

        def time_range(self):
            return (
                faery.Time(microseconds=int(events["t"][0])),
                faery.Time(microseconds=int(events["t"][-1]) + 1),
            )

    chunks = _chunks(Packets(), chunk_length)
    numpy.testing.assert_array_equal(numpy.concatenate(chunks), events)
    assert all(len(chunk) == chunk_length for chunk in chunks[:-1])
    assert 0 < len(chunks[-1]) <= chunk_length
