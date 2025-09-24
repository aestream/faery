# Arrays and static inputs

Faery supports creating event and frame streams from programmatically generated data, making it useful for testing, simulations, and synthetic data generation.

## Event streams from arrays

Create event streams from NumPy arrays containing pre-computed event data.

### Python API

```python
import faery
import numpy as np

# Create synthetic events
events = np.array([
    (1000, 100, 200, True),   # t=1000μs, x=100, y=200, polarity=True
    (2000, 150, 250, False),  # t=2000μs, x=150, y=250, polarity=False
    (3000, 200, 300, True),   # t=3000μs, x=200, y=300, polarity=True
], dtype=faery.EVENTS_DTYPE)

# Create stream from array
stream = faery.events_stream_from_array(
    events=events,
    dimensions=(640, 480)
)

# Use like any other stream
stream.render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.starry_night) \
      .to_file("synthetic.mp4")
```

### Event data format

Events must use Faery's standard dtype:

```python
import numpy as np
import faery

# Standard event format
EVENTS_DTYPE = np.dtype([
    ("t", "<u8"),      # Timestamp (microseconds, uint64)
    ("x", "<u2"),      # X coordinate (uint16)
    ("y", "<u2"),      # Y coordinate (uint16)
    (("p", "on"), "?") # Polarity (bool)
])

# Create events with correct dtype
events = np.zeros(100, dtype=faery.EVENTS_DTYPE)
events["t"] = np.arange(0, 100000, 1000)  # Events every 1ms
events["x"] = np.random.randint(0, 640, 100)
events["y"] = np.random.randint(0, 480, 100)
events["p"] = np.random.choice([True, False], 100)

stream = faery.events_stream_from_array(events, dimensions=(640, 480))
```

## Frame streams from lists

Create frame streams from pre-computed image sequences.

### Python API

```python
import faery
import numpy as np

# Generate synthetic frames
frames = []
for i in range(60):  # 60 frames
    # Create a moving bright spot
    frame = np.zeros((480, 640), dtype=np.uint8)
    x = int(320 + 100 * np.sin(i * 0.1))
    y = int(240 + 100 * np.cos(i * 0.1))
    frame[y-5:y+5, x-5:x+5] = 255
    frames.append(frame)

# Create frame stream
stream = faery.frame_stream_from_list(
    start_t=0 * faery.us,       # Start at t=0
    frequency_hz=30.0,          # 30 FPS
    frames=frames
)

# Convert to video
stream.to_file("synthetic_frames.mp4")
```

## Frame streams from functions

Generate frames procedurally using functions.

### Python API

```python
import faery
import numpy as np

def generate_frame(t: faery.Time) -> np.ndarray:
    """Generate a frame based on timestamp"""
    # Convert timestamp to seconds
    time_s = t.seconds()

    # Create animated pattern
    frame = np.zeros((480, 640), dtype=np.uint8)

    # Moving circle
    center_x = int(320 + 200 * np.sin(time_s * 2))
    center_y = int(240 + 100 * np.cos(time_s * 3))

    y, x = np.ogrid[:480, :640]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    frame[distance < 30] = 255

    return frame

# Create procedural frame stream
stream = faery.frame_stream_from_function(
    start_t=0 * faery.us,
    frequency_hz=24.0,
    dimensions=(640, 480),
    frame_count=120,  # 5 seconds at 24 FPS
    get_frame=generate_frame
)

stream.to_file("procedural_animation.mp4")
```

## Synthetic event patterns

### Moving objects

```python
import faery
import numpy as np

def create_moving_object_events(
    start_x: int, start_y: int,
    velocity_x: float, velocity_y: float,
    dimensions: tuple[int, int],
    duration_us: int,
    object_size: int = 5
) -> np.ndarray:
    """Create events for a moving object"""
    events = []

    for t in range(0, duration_us, 1000):  # Every 1ms
        # Current position
        x = int(start_x + velocity_x * t / 1000000)  # velocity in pixels/second
        y = int(start_y + velocity_y * t / 1000000)

        # Create events around object position
        for dx in range(-object_size, object_size + 1):
            for dy in range(-object_size, object_size + 1):
                px, py = x + dx, y + dy
                if 0 <= px < dimensions[0] and 0 <= py < dimensions[1]:
                    # Random polarity for texture
                    polarity = np.random.choice([True, False])
                    events.append((t, px, py, polarity))

    return np.array(events, dtype=faery.EVENTS_DTYPE)

# Create moving object
events = create_moving_object_events(
    start_x=100, start_y=240,
    velocity_x=200,  # 200 pixels/second to the right
    velocity_y=50,   # 50 pixels/second downward
    dimensions=(640, 480),
    duration_us=3000000  # 3 seconds
)

stream = faery.events_stream_from_array(events, dimensions=(640, 480))
```

### Noise patterns

```python
def create_noise_events(
    dimensions: tuple[int, int],
    duration_us: int,
    event_rate_hz: float
) -> np.ndarray:
    """Create random noise events"""
    total_events = int(event_rate_hz * duration_us / 1000000)

    events = np.zeros(total_events, dtype=faery.EVENTS_DTYPE)
    events["t"] = np.sort(np.random.randint(0, duration_us, total_events))
    events["x"] = np.random.randint(0, dimensions[0], total_events)
    events["y"] = np.random.randint(0, dimensions[1], total_events)
    events["p"] = np.random.choice([True, False], total_events)

    return events

# Create noise background
noise = create_noise_events(
    dimensions=(640, 480),
    duration_us=5000000,  # 5 seconds
    event_rate_hz=10000   # 10k events/second
)

stream = faery.events_stream_from_array(noise, dimensions=(640, 480))
```

### Periodic patterns

```python
def create_periodic_flash(
    center: tuple[int, int],
    radius: int,
    period_us: int,
    cycles: int,
    dimensions: tuple[int, int]
) -> np.ndarray:
    """Create periodic flashing pattern"""
    events = []
    cx, cy = center

    for cycle in range(cycles):
        # Flash on
        flash_time = cycle * period_us
        y, x = np.ogrid[:dimensions[1], :dimensions[0]]
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        flash_pixels = np.where(distance < radius)

        for px, py in zip(flash_pixels[1], flash_pixels[0]):
            events.append((flash_time, px, py, True))

        # Flash off (half period later)
        off_time = flash_time + period_us // 2
        for px, py in zip(flash_pixels[1], flash_pixels[0]):
            events.append((off_time, px, py, False))

    return np.array(events, dtype=faery.EVENTS_DTYPE)

# Create blinking circle
flash_events = create_periodic_flash(
    center=(320, 240),
    radius=50,
    period_us=500000,  # 500ms period (2 Hz)
    cycles=10,
    dimensions=(640, 480)
)

stream = faery.events_stream_from_array(flash_events, dimensions=(640, 480))
```

## Testing and validation

### Unit test events

```python
import faery
import numpy as np

# Create minimal test case
test_events = np.array([
    (0, 0, 0, True),
    (1000, 639, 479, False),  # Test corner cases
], dtype=faery.EVENTS_DTYPE)

stream = faery.events_stream_from_array(test_events, dimensions=(640, 480))

# Verify stream properties
assert stream.dimensions() == (640, 480)

# Test processing pipeline
processed = stream.crop(10, 10, 620, 460).take(100)
# ... additional validation
```

### Performance benchmarking

```python
import time
import faery
import numpy as np

# Create large synthetic dataset
num_events = 1000000
events = np.zeros(num_events, dtype=faery.EVENTS_DTYPE)
events["t"] = np.arange(num_events) * 10  # 10μs intervals
events["x"] = np.random.randint(0, 1280, num_events)
events["y"] = np.random.randint(0, 720, num_events)
events["p"] = np.random.choice([True, False], num_events)

# Benchmark processing speed
start_time = time.time()
stream = faery.events_stream_from_array(events, dimensions=(1280, 720))
result = stream.regularize(frequency_hz=60.0).take(1000)
elapsed = time.time() - start_time

print(f"Processed {num_events} events in {elapsed:.3f}s")
print(f"Processing rate: {num_events/elapsed:.0f} events/second")
```

## Integration with other libraries

### From OpenCV

```python
import cv2
import numpy as np
import faery

# Load video and convert to events
cap = cv2.VideoCapture('input_video.mp4')
events = []
prev_frame = None
timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is not None:
        # Simple change detection
        diff = cv2.absdiff(gray, prev_frame)
        y_coords, x_coords = np.where(diff > 30)  # Threshold

        for x, y in zip(x_coords, y_coords):
            polarity = gray[y, x] > prev_frame[y, x]
            events.append((timestamp, x, y, polarity))

    prev_frame = gray
    timestamp += 33333  # ~30 FPS (33.33ms intervals)

cap.release()

# Convert to Faery stream
events_array = np.array(events, dtype=faery.EVENTS_DTYPE)
stream = faery.events_stream_from_array(events_array, dimensions=(640, 480))
```

### From simulation frameworks

```python
# Example: Convert simulation data to Faery format
def simulation_to_events(sim_data, dimensions):
    """Convert simulation output to event format"""
    events = []

    for time_step, positions, polarities in sim_data:
        timestamp_us = int(time_step * 1000000)  # Convert to microseconds

        for (x, y), polarity in zip(positions, polarities):
            if 0 <= x < dimensions[0] and 0 <= y < dimensions[1]:
                events.append((timestamp_us, int(x), int(y), bool(polarity)))

    return np.array(events, dtype=faery.EVENTS_DTYPE)

# Usage with hypothetical simulation
# sim_results = run_my_simulation(...)
# events = simulation_to_events(sim_results, (640, 480))
# stream = faery.events_stream_from_array(events, dimensions=(640, 480))
```

## Common patterns

### Combining multiple sources

```python
import numpy as np
import faery

# Create multiple event sources
source1 = create_moving_object_events(100, 200, 50, 25, (640, 480), 2000000)
source2 = create_noise_events((640, 480), 2000000, 5000)
source3 = create_periodic_flash((500, 300), 30, 200000, 10, (640, 480))

# Combine and sort by timestamp
all_events = np.concatenate([source1, source2, source3])
sorted_indices = np.argsort(all_events["t"])
combined_events = all_events[sorted_indices]

# Create unified stream
stream = faery.events_stream_from_array(combined_events, dimensions=(640, 480))
```

This approach allows you to create complex synthetic scenes with multiple objects, noise, and patterns for comprehensive testing and algorithm development.
