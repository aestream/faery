# Input sources

Faery supports multiple input sources, each creating different types of streams. Understanding which input creates which stream type is crucial for building effective processing pipelines.

## Stream type summary

| Input Source | Stream Type | Finite? | Regular? | Use Cases |
|--------------|-------------|---------|----------|-----------|
| **Files** | `FiniteEventsStream` | ✅ Yes | ❌ No | Batch processing, analysis, reproducible results |
| **Event Cameras** | `EventsStream` | ❌ No | ❌ No | Real-time processing, live monitoring |
| **UDP Streams** | `EventsStream` | ❌ No | ❌ No | Network integration, distributed processing |
| **Arrays (Python)** | `FiniteEventsStream` | ✅ Yes | ❌ No | Testing, simulation, synthetic data |
| **Standard Input** | `FiniteEventsStream` | ✅ Yes | ❌ No | Pipeline integration, shell scripting |

## Input source details

### 📁 Files
**Stream Type**: `FiniteEventsStream` (Finite + Irregular)

**Supported Formats**: AEDAT4, ES, Prophesee RAW, DAT, EVT, CSV

**Characteristics**:
- Known duration and size
- Can be converted to arrays with `.to_array()`
- Perfect for video generation after regularization
- Reproducible processing

**Example**:
```python
# File → Finite → Regular → Video
faery.events_stream_from_file("data.aedat4") \
    .regularize(frequency_hz=30.0) \
    .render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.devon) \
    .to_file("output.mp4")
```

### 📷 Event Cameras
**Stream Type**: `EventsStream` (Infinite + Irregular)

**Supported Hardware**: DVXplorer, DAVIS346, EVK4, EVK3 HD

**Characteristics**:
- Continuous data stream
- Requires `.time_slice()` or `.event_slice()` for video output
- Real-time processing capabilities
- Live monitoring and streaming

**Example**:
```python
# Camera → Slice → Finite → Regular → Video
faery.events_stream_from_camera() \
    .time_slice(0 * faery.s, 10 * faery.s) \
    .regularize(frequency_hz=30.0) \
    .render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.starry_night) \
    .to_file("camera_recording.mp4")
```

### 🌐 UDP Streams
**Stream Type**: `EventsStream` (Infinite + Irregular)

**Supported Formats**: t64_x16_y16_on8, t32_x16_y15_on1

**Characteristics**:
- Network-based event streaming
- Low-latency for distributed systems
- Requires dimensions specification
- Can be finite or infinite depending on sender

**Example**:
```python
# UDP → Process → Stream to another UDP endpoint
faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("localhost", 7777)
).remove_off_events() \
 .to_udp(("processing-server", 8888))
```

### 🔢 Arrays (Python)
**Stream Type**: `FiniteEventsStream` (Finite + Irregular)

**Data Source**: NumPy arrays with `faery.EVENTS_DTYPE`

**Characteristics**:
- Programmatically generated data
- Perfect for testing and validation
- Known size and duration
- Synthetic data generation

**Example**:
```python
# Array → Direct video generation
events = create_synthetic_events()  # Your function
faery.events_stream_from_array(events, dimensions=(640, 480)) \
    .regularize(frequency_hz=24.0) \
    .render(decay="exponential", tau="00:00:00.100000", colormap=faery.colormaps.batlow) \
    .to_file("synthetic.mp4")
```

### 📥 Standard Input
**Stream Type**: `FiniteEventsStream` (Finite + Irregular)

**Supported Format**: CSV only

**Characteristics**:
- Command-line pipeline integration
- Finite by nature (piped data has an end)
- Configurable CSV parsing
- Shell scripting integration

**Example**:
```bash
# Shell pipeline → Faery processing
cat events.csv | faery input stdin --dimensions 640x480 \
    filter regularize 30.0 \
    filter render exponential "00:00:00.200000" starry_night \
    output file processed.mp4
```
