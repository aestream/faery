from __future__ import annotations

import collections.abc
import importlib.util
import logging
import typing

import numpy as np

from faery import events_stream

logger = logging.getLogger(__name__)


def has_event_camera_drivers():
    return importlib.util.find_spec("event_camera_drivers") is not None


def has_neuromorphic_drivers():
    return importlib.util.find_spec("neuromorphic_drivers") is not None


class EventCameraDriverStream(events_stream.EventsStream):
    def __init__(
        self,
        manufacturer: typing.Literal["Prophesee", "Inivation"] | None = None,
        buffer_size: int = 1024,
    ):
        """Create an events stream using the event-camera-drivers library

        Args:
            manufacturer (Optional[typing.Literal["Prophesee", "Inivation"]]): Camera manufacturer.
                Defaults to automatic detection which might take some time.
            buffer_size (int): The size of the buffer to use for the event stream, defaults to 1024

        Returns:
            An (infinite) event stream from the camera

        Usage:
            >>> stream = EventCameraStream() # Open camera (will fail if no camera is connected)
            >>> stream.map(...)              # Use the stream as any other (infinite) event stream
        """
        try:
            import event_camera_drivers as evd  # type: ignore

            self.camera = evd.InivationCamera(buffer_size=buffer_size)
        except ImportError as error:
            raise ImportError(
                "event_camera_drivers is not installed, run `pip install event-camera-drivers`"
            ) from error
        except ValueError as error:
            raise ValueError("no camera found using libcaer") from error

    def __iter__(self) -> collections.abc.Iterator[np.ndarray]:
        logger.info("starting streaming from event_camera_drivers: %s", self.camera)
        while self.camera.is_running():
            v = next(self.camera)
            yield v

    def dimensions(self):
        return self.camera.resolution()


class NeuromorphicCameraStream(events_stream.EventsStream):
    def __init__(self):
        try:
            import neuromorphic_drivers as nd  # type: ignore

            self.nd = nd
            self.device_list = nd.list_devices()
            if len(self.device_list) == 0:
                raise RuntimeError(
                    "No event camera found, did you plug it in and install the udev rules?"
                )
        except ImportError as error:
            raise ImportError(
                "neuromorphic_drivers is not installed, run `pip install neuromorphic-drivers`"
            ) from error

    def __iter__(self) -> collections.abc.Iterator[np.ndarray]:
        logger.info(
            "starting streaming from neuromorphic_drivers: %s", self.device_list[0]
        )
        with self.nd.open() as device:
            for status, packet in device:
                # TODO: Check status
                events = packet.polarity_events
                if events is not None:
                    # Neuromorphic drivers use similar dtype, so we can safely cast
                    yield events.astype(events_stream.EVENTS_DTYPE)

    def dimensions(self):
        # TODO: Use the current camera, rather than device_list[0]
        if self.device_list[0].name == self.nd.generated.enums.Name.INIVATION_DVXPLORER:
            return (640, 480)
        elif (
            self.device_list[0].name == self.nd.generated.enums.Name.PROPHESEE_EVK4
            or self.device_list[0].name
            == self.nd.generated.enums.Name.PROPHESEE_EVK3_HD
        ):
            return (1280, 720)
        elif (
            self.device_list[0].name == self.nd.generated.enums.Name.INIVATION_DAVIS346
        ):
            return (346, 260)
        else:
            raise ValueError("Unknown Camera", self.device_list[0].name)


def events_stream_from_camera(
    driver: typing.Literal["EventCameraDrivers", "NeuromorphicDrivers", "Auto"]
    | None = None,
    manufacturer: typing.Literal["Inivation", "Prophesee"] | None = None,
    buffer_size: int = 1024,
):
    stream = None
    error = None
    if driver is None or driver == "EventCameraDrivers" or driver == "Auto":
        try:
            stream = EventCameraDriverStream(
                manufacturer=manufacturer, buffer_size=buffer_size
            )
        except Exception as e:  # noqa: BLE001
            error = e
    if driver is None or driver == "NeuromorphicDrivers" or driver == "Auto":
        try:
            stream = NeuromorphicCameraStream()
        except Exception as e:  # noqa: BLE001
            error = e

    if stream is None:
        raise ValueError("No Event Camera found:", error)

    return stream
