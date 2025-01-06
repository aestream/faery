import importlib.util
import .events_stream as events_stream

def has_event_camera_drivers():
    return importlib.util.find_spec("event_camera_drivers") is not None

def has_inivation_camera_drivers():
    if are_event_camera_drivers_available():
        import event_camera_drivers as evd
        return hasattr(evd, "InivationCamera")
    return False

if has_inivation_camera_drivers():
    import event_camera_drivers as evd

    class InivationCameraStream(events_stream.EventsStream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def events_stream_from_inivation_camera():
        """Create an events stream from a connected Inivation camera

        Returns:
            An event stream from the camera
        """
        return event_camera_drivers.

