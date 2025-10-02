# Event cameras inputs

Faery supports direct streaming from event cameras, enabling real-time data processing and visualization. Note that this requires additional driver installations.

## Supported cameras

Faery supports event cameras from two major manufacturers through different driver systems:

### Inivation cameras
- **DVXplorer** (640×480)
- **DAVIS346** (346×260)
- Requires: [`event_camera_drivers`](https://github.com/aestream/event-camera-drivers/) or [`neuromorphic_drivers`](https://github.com/neuromorphicsystems/neuromorphic-drivers)

### Prophesee cameras
- **EVK4** (1280×720)
- **EVK3 HD** (1280×720)
- Requires: [`neuromorphic_drivers`](https://github.com/neuromorphicsystems/neuromorphic-drivers)

## Driver installation

Faery supports two driver systems. Install at least one:

### Option 1: event_camera_drivers
```sh
pip install event_camera_drivers
```
- Works well with Inivation cameras
- Uses existing libcaer drivers under the hood

### Option 2: neuromorphic_drivers
```sh
pip install neuromorphic_drivers
```
- Custom drivers
- Supports both Inivation and Prophesee cameras

## Usage

### Command line

```sh
# Stream from any detected camera to file
faery input inivation camera output file recording.es

# Stream to UDP for real-time processing
faery input inivation camera output udp localhost:7777

# Create real-time visualization
faery input inivation camera filter regularize 30.0 filter render exponential "00:00:00.100000" starry_night output mp4 live.mp4
```

### Python

```python
import faery

# Auto-detect and connect to camera
stream = faery.events_stream_from_camera()

# Specify driver system
stream = faery.events_stream_from_camera(driver="EventCameraDrivers")
stream = faery.events_stream_from_camera(driver="NeuromorphicDrivers")

# Specify manufacturer (helps with faster detection)
stream = faery.events_stream_from_camera(manufacturer="Inivation")
stream = faery.events_stream_from_camera(manufacturer="Prophesee")

# Configure buffer size for performance tuning
stream = faery.events_stream_from_camera(buffer_size=2048)

# Use in a processing pipeline
faery.events_stream_from_camera() \
    .time_slice(0 * faery.s, 10 * faery.s) \
    .regularize(frequency_hz=30.0) \
    .render(tau="00:00:00.002000", decay="exponential", colormap=faery.colormaps.starry_night) \
    .to_file("live.mp4")
```

## Camera detection and selection

Faery automatically detects available cameras and drivers:

```python
# Check driver availability
import faery.event_camera_input as eci

if eci.has_event_camera_drivers():
    print("event_camera_drivers available")

if eci.has_neuromorphic_drivers():
    print("neuromorphic_drivers available")

# Auto-detection tries drivers in order of preference
stream = faery.events_stream_from_camera(driver="Auto")  # Default behavior
```

## Configuration options

### Buffer size
Controls the internal buffer size for event streaming:

```python
# Larger buffers reduce latency spikes but use more memory
stream = faery.events_stream_from_camera(buffer_size=4096)  # Default: 1024
```

### Manufacturer hint
Speeds up camera detection by specifying the expected manufacturer:

```python
# Skip detection time by specifying manufacturer
stream = faery.events_stream_from_camera(manufacturer="Inivation")
```

## Real-time processing examples

### Live visualization
```python
import faery

# Record 10 seconds of camera data as video
faery.events_stream_from_camera() \
    .time_slice(0 * faery.s, 10 * faery.s) \
    .regularize(frequency_hz=30.0) \
    .render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.starry_night) \
    .to_file("live.mp4")
```

### Network streaming
```python
# Stream events over UDP for distributed processing
faery.events_stream_from_camera() \
    .to_udp(("localhost", 7777))
```


## Troubleshooting

### No camera detected
```
ValueError: No Event Camera found
```
**Solutions:**
1. Check physical camera connection (USB)
2. Verify driver installation: `pip list | grep -E "(event_camera|neuromorphic)"`
3. For neuromorphic_drivers: ensure udev rules are installed
4. Try specifying manufacturer: `manufacturer="Inivation"`

### Permission issues (Linux)
```
PermissionError: [Errno 13] Permission denied
```
**Solutions:**
1. Install udev rules for your camera - see the [guide on NeuromorphicDrivers' GitHub](https://github.com/neuromorphicsystems/neuromorphic-drivers?tab=readme-ov-file#udev-rules)
2. Add user to appropriate groups (dialout, plugdev)
3. Restart or re-login after group changes

### Driver import errors
```
ImportError: No module named 'event_camera_drivers' / 'neuromorphic_drivers'
```
**Solutions:**
1. Install missing driver: `pip install event_camera_drivers` or `pip install neuromorphic_drivers`
2. Check virtual environment activation
3. Verify compatible Python version

## Hardware requirements

- **USB 3.0 or higher** recommended for high-resolution cameras
- **Available USB ports** (cameras may require significant bandwidth)
- **Sufficient RAM** for buffering (especially at high event rates)
- **Linux/Windows/macOS** support varies by driver system. Please submit issues in the respective driver repositories
