import importlib.util
import logging
import typing

import numpy as np

import faery.events_stream as events_stream

def has_event_camera_drivers():
    return importlib.util.find_spec("event_camera_drivers") is not None


def has_inivation_camera_drivers():
    if has_event_camera_drivers():
        import event_camera_drivers as evd
        return hasattr(evd, "InivationCamera")
    return False


class InivationCameraStream(events_stream.EventsStream):
    def __init__(self, buffer_size: int = 1024):
        """Create an events stream from a connected Inivation camera

        Args:
            buffer_size: The size of the buffer to use for the event stream, defaults to 1024

        Returns:
            An (infinite) event stream from the camera

        Usage:
            >>> stream = InivationCameraStream() # Open camera (will fail if no camera is connected)
            >>> stream.map(...)                  # Use the stream as any other (infinite) event stream
        """

        super().__init__()
        
        try:
            import event_camera_drivers as evd
            self.camera = evd.InivationCamera(buffer_size=buffer_size)
        except ImportError as e:
            logging.error("Inivation camera drivers not available, please install the event_camera_drivers library")
            raise e
        
    def __iter__(self) -> typing.Iterator[np.ndarray]:
        while self.camera.is_running():
            v = next(self.camera)
            yield v

    def dimensions(self):
        return self.camera.resolution()


def events_stream_from_camera(manufacturer: typing.Literal["Inivation", "Prophesee"], buffer_size: int = 1024):
    if manufacturer == "Inivation":
        return InivationCameraStream(buffer_size)
    elif manufacturer == "Prophesee":
        raise NotImplementedError("Prophesee camera drivers are not implemented yet")
    else:
        raise ValueError(f"Unknown camera manufacturer: {manufacturer}")

