# Network (UDP) inputs

Faery supports reading event streams from UDP network sources, enabling real-time distributed processing and camera streaming over networks.

## When to use UDP input

UDP streaming is useful for:
- **Real-time applications**: Processing events as they arrive from cameras or other sources
- **Distributed systems**: Receiving events from remote cameras or processing nodes
- **Low-latency pipelines**: Minimal protocol overhead for time-critical applications
- **Network integration**: Connecting Faery to existing UDP-based event systems

## Supported UDP formats

Faery supports two binary UDP formats for event transmission:

### t64_x16_y16_on8 (Default)
- **Timestamp**: 64-bit (8 bytes)
- **X coordinate**: 16-bit (2 bytes)
- **Y coordinate**: 16-bit (2 bytes)
- **Polarity**: 8-bit (1 byte)
- **Total**: 13 bytes per event
- **Use case**: High precision timestamps, standard coordinate resolution

### t32_x16_y15_on1
- **Timestamp**: 32-bit (4 bytes)
- **X coordinate**: 16-bit (2 bytes)
- **Y coordinate**: 15-bit + 1-bit polarity (2 bytes total)
- **Total**: 8 bytes per event
- **Use case**: Bandwidth-constrained networks, lower timestamp precision acceptable

## Basic usage

### Command line

```sh
# Listen on localhost port 7777 (IPv4)
faery input udp localhost:7777 --dimensions 640x480 output file received.es

# Specify UDP format
faery input udp localhost:7777 --dimensions 640x480 --format t32_x16_y15_on1 output file received.es

# IPv6 address
faery input udp [::1]:7777 --dimensions 640x480 output file received.es

# Listen and stream to another UDP endpoint
faery input udp localhost:7777 --dimensions 640x480 output udp remote-host:8888
```

### Python

```python
import faery

# Basic UDP input
stream = faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("localhost", 7777)
)

# Specify format
stream = faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("localhost", 7777),
    format="t32_x16_y15_on1"
)

# IPv6
stream = faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("::1", 7777, None, None)  # IPv6 format
)

# Use in processing pipeline
faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("localhost", 7777)
).time_slice(0 * faery.s, 10 * faery.s) \
 .regularize(frequency_hz=30.0) \
 .render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.starry_night) \
 .to_file("network_stream.mp4")
```

## Real-time processing examples

### Camera to network pipeline
```python
# Send camera data over UDP
faery.events_stream_from_camera() \
    .to_udp("remote-host", 7777)

# Receive and process on remote machine
faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("0.0.0.0", 7777)  # Listen on all interfaces
).regularize(frequency_hz=60.0) \
 .render(decay="exponential", tau="00:00:00.100000", colormap=faery.colormaps.starry_night) \
 .to_file("remote_camera.mp4")
```

### Multi-hop processing
```python
# Stage 1: Basic filtering
faery.events_stream_from_udp(
    dimensions=(1280, 720),
    address=("localhost", 7777)
).remove_off_events() \
 .to_udp(("processing-node", 8888))

# Stage 2: Advanced processing
faery.events_stream_from_udp(
    dimensions=(1280, 720),
    address=("0.0.0.0", 8888)
).regularize(frequency_hz=30.0) \
 .render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.devon) \
 .to_file("processed_output.mp4")
```

### Event rate monitoring
```python
# Monitor network event rates
faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("localhost", 7777)
).to_event_rate(window_duration_us=1000000) \
 .to_file("network_activity.csv")
```

## Firewall settings

Your firewall can sometimes block UDP traffic, so ensure that your system opens the right ports.
Below are examples for Linux and Windows:

```sh
# Allow UDP input on port 7777 (Linux iptables)
sudo iptables -A INPUT -p udp --dport 7777 -j ACCEPT

# Windows Firewall (PowerShell)
New-NetFirewallRule -DisplayName "Faery UDP" -Direction Inbound -Protocol UDP -LocalPort 7777 -Action Allow
```

## Troubleshooting

### Connection issues

**Address already in use**
```
OSError: [Errno 98] Address already in use
```
- Check if another process is using the port: `netstat -ulnp | grep 7777`
- Try a different port

**No route to host**
```
OSError: [Errno 113] No route to host
```
- Verify network connectivity: `ping target-host`
- Check firewall rules on both sender and receiver
- Ensure correct IP address and port

### Data issues

**No events received**
Either dump events to stdout or write a custom loop to check that you're actually seeing data.
```python
# Check if data is arriving (will block until events received)
for events in stream:
    print(f"Received {len(events)} events")
    break  # Exit after first packet
```

**Malformed events**
- Verify sender and receiver use same UDP format
- Check network MTU settings (large packets may be fragmented)
- Ensure sender uses correct event encoding

**High latency**
- Reduce processing complexity in pipeline
- Check network latency: `ping -c 10 target-host`
- Consider using faster UDP format (t32_x16_y15_on1)

### Performance optimization

**High CPU usage**
```python
# Add regularization to reduce processing frequency
stream = faery.events_stream_from_udp(
    dimensions=(640, 480),
    address=("localhost", 7777)
).regularize(frequency_hz=30.0)  # Limit to 30 FPS processing
```

**Memory usage**
- Events are processed in streaming fashion (low memory footprint)
- For finite processing, consider `.take()` to limit event count
- Monitor with `top` or `htop` for actual memory usage

**Network bandwidth**
```python
# Monitor received data rate
import time
start_time = time.time()
event_count = 0

for events in stream:
    event_count += len(events)
    elapsed = time.time() - start_time
    if elapsed > 5.0:  # Report every 5 seconds
        rate = event_count / elapsed
        print(f"Receiving {rate:.0f} events/second")
        event_count = 0
        start_time = time.time()
```
