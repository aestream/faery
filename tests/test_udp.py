import pytest
import threading
import time

import numpy as np
import faery


@pytest.mark.parametrize(
    "format_type,payload_length,num_events,port",
    [
        ("t64_x16_y16_on8", 1209, 500, 29999),
        ("t32_x16_y15_on1", 1208, 500, 29998),
    ],
)
def test_udp_encoder_decoder(format_type, payload_length, num_events, port):
    """Test UDP encoder/decoder with random events."""

    dimensions = (640, 480)
    address = ("localhost", port)

    # Create random test events
    test_events = np.zeros(num_events, dtype=faery.EVENTS_DTYPE)
    test_events["t"] = np.arange(0, num_events * 1000, 1000, dtype=np.uint64)
    test_events["x"] = np.random.randint(0, dimensions[0], num_events, dtype=np.uint16)
    test_events["y"] = np.random.randint(0, dimensions[1], num_events, dtype=np.uint16)
    test_events["on"] = np.random.randint(0, 2, num_events, dtype=bool)

    received_events = []
    stop_receiver = threading.Event()
    receiver_exception = None

    def receiver():
        """Receive events from UDP."""
        nonlocal receiver_exception
        try:
            stream = faery.events_stream_from_udp(
                dimensions=dimensions,
                address=address,
                format=format_type
            )
            for events in stream:
                received_events.append(events.copy())
                if stop_receiver.is_set():
                    break
        except Exception as e:
            receiver_exception = e

    def sender():
        """Send test events via UDP."""
        time.sleep(0.2)  # Give receiver time to start listening
        stream = faery.events_stream_from_array(test_events, dimensions)
        stream.to_udp(
            address,
            format=format_type,
            payload_length=payload_length
        )
        time.sleep(0.1)  # Give receiver time to collect all packets
        stop_receiver.set()

    # Start receiver and sender threads
    receiver_thread = threading.Thread(target=receiver, daemon=True)
    sender_thread = threading.Thread(target=sender, daemon=True)

    receiver_thread.start()
    sender_thread.start()

    # Wait for completion with timeout
    sender_thread.join(timeout=5)
    receiver_thread.join(timeout=5)

    # Verify no exceptions occurred
    assert receiver_exception is None, f"Receiver exception: {receiver_exception}"

    # Verify event count
    all_received = np.concatenate(received_events)
    assert len(all_received) == num_events, (
        f"Expected {num_events} events, got {len(all_received)}"
    )

    # Verify event data matches (with consideration for timestamp wraparound in t32 format)
    if format_type == "t64_x16_y16_on8":
        # For t64 format, all fields should match exactly
        assert np.array_equal(all_received["x"], test_events["x"]), "X coordinates don't match"
        assert np.array_equal(all_received["y"], test_events["y"]), "Y coordinates don't match"
        assert np.array_equal(all_received["on"], test_events["on"]), "Polarity doesn't match"
        assert np.array_equal(all_received["t"], test_events["t"]), "Timestamps don't match"
    else:
        # For t32 format, timestamps are 32-bit so we check modulo 2^32
        assert np.array_equal(all_received["x"], test_events["x"]), "X coordinates don't match"
        assert np.array_equal(all_received["y"], test_events["y"]), "Y coordinates don't match"
        assert np.array_equal(all_received["on"], test_events["on"]), "Polarity doesn't match"
        # Check timestamps modulo 2^32
        assert np.array_equal(
            all_received["t"] & 0xFFFFFFFF,
            test_events["t"] & 0xFFFFFFFF
        ), "Timestamps don't match"

