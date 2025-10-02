# Standard input (stdin)

Faery supports reading event data from standard input, enabling integration with command-line pipelines and shell scripting workflows.

## When to use stdin input

Standard input is useful for:
- **Unix pipeline integration**: Chaining multiple command-line tools
- **Shell scripting**: Processing data streams in batch scripts
- **Data preprocessing**: Filtering or transforming data before Faery processing
- **Remote processing**: Receiving data through SSH pipes or network commands

## Supported format

Currently, stdin input only supports **CSV format**. The data must be structured as comma-separated values with configurable column mapping.

## Basic usage

### Command line

```sh
# Read CSV from stdin and convert to file
cat events.csv | faery input stdin --dimensions 640x480 output file output.es

# Process data from another command
generate_events.py | faery input stdin --dimensions 1280x720 output file output.aedat4

# Custom CSV format
cat custom_events.csv | faery input stdin --dimensions 640x480 \
    --csv-separator ";" \
    --csv-t-index 0 \
    --csv-x-index 1 \
    --csv-y-index 2 \
    --csv-p-index 3 \
    output file output.es

# Skip header row
tail -n +2 events_with_header.csv | faery input stdin --dimensions 640x480 \
    --no-csv-has-header \
    output file output.es
```

### Python

```python
import faery
import sys

# Read from stdin (CSV format only)
stream = faery.events_stream_from_stdin(
    dimensions=(640, 480)
)

# Custom CSV properties
csv_props = faery.CsvProperties(
    has_header=False,
    separator=";",
    t_index=0,
    x_index=1,
    y_index=2,
    p_index=3
)

stream = faery.events_stream_from_stdin(
    dimensions=(640, 480),
    csv_properties=csv_props,
    t0=1000000 * faery.us  # Start time offset
)

# Process the stream
stream.regularize(frequency_hz=30.0) \
      .render(decay="exponential", tau="00:00:00.200000", colormap=faery.colormaps.starry_night) \
      .to_file("stdin_output.mp4")
```

## CSV format requirements

### Default format
```csv
t,x,y,p
1000,100,200,1
2000,150,250,0
3000,200,300,1
```

- **Column 0 (t)**: Timestamp in microseconds
- **Column 1 (x)**: X coordinate
- **Column 2 (y)**: Y coordinate
- **Column 3 (p)**: Polarity (0/1 or True/False)

### Custom column mapping

Configure which columns contain which data:

```sh
# Different column order: x,y,t,p
echo "150,200,1000,1" | faery input stdin --dimensions 640x480 \
    --csv-x-index 0 \
    --csv-y-index 1 \
    --csv-t-index 2 \
    --csv-p-index 3 \
    output file output.es
```

### Alternative separators

```sh
# Tab-separated values
cat events.tsv | faery input stdin --dimensions 640x480 \
    --csv-separator $'\t' \
    output file output.es

# Semicolon-separated
cat events.csv | faery input stdin --dimensions 640x480 \
    --csv-separator ";" \
    output file output.es
```

## Pipeline integration examples

### Data preprocessing

```sh
# Filter events by polarity before processing
awk -F, '$4==1' events.csv | faery input stdin --dimensions 640x480 \
    --no-csv-has-header \
    output file positive_events.es

# Select time range
awk -F, '$1>=1000000 && $1<=2000000' events.csv | \
    faery input stdin --dimensions 640x480 \
    --no-csv-has-header \
    output file time_slice.es
```

### Multi-stage processing

```sh
# Stage 1: Preprocess with custom script
cat raw_data.txt | ./preprocess.py | \
# Stage 2: Convert to Faery format
faery input stdin --dimensions 1280x720 output file intermediate.es

# Stage 3: Render video
faery input file intermediate.es \
    filter regularize 30.0 \
    filter render exponential 0.2 devon \
    output file final.mp4
```

### Network integration

```sh
# Receive data over SSH and process
ssh remote-host "cat /path/to/events.csv" | \
    faery input stdin --dimensions 640x480 output file remote_events.es

# Process streaming data from network service
curl -s "http://api.example.com/events.csv" | \
    faery input stdin --dimensions 640x480 \
    filter regularize 60.0 \
    output udp localhost 7777
```

### Database integration

```sh
# Query database and process results
psql -d events_db -c "SELECT timestamp_us, x, y, polarity FROM events WHERE timestamp_us > 1000000" \
    --csv --no-align --field-separator=',' | \
    tail -n +2 | \
    faery input stdin --dimensions 1280x720 --no-csv-has-header output file db_events.es

# MySQL export
mysql -u user -p -D events_db -e "SELECT t, x, y, p FROM events" \
    --batch --raw --silent | \
    faery input stdin --dimensions 640x480 --csv-separator $'\t' --no-csv-has-header \
    output file mysql_events.es
```

## Real-time stream processing

### Continuous processing

```sh
# Process events as they arrive (named pipe)
mkfifo event_pipe
./event_generator > event_pipe &
faery input stdin --dimensions 640x480 < event_pipe \
    filter regularize 30.0 \
    filter render exponential 0.1 starry_night \
    output file live_stream.mp4
```

### Log file monitoring

```sh
# Process new events from growing log file
tail -f /var/log/events.csv | \
    faery input stdin --dimensions 640x480 --no-csv-has-header \
    filter event_rate 100000 \
    output file event_rates.csv
```

## Error handling and validation

### Input validation

```sh
# Validate CSV format before processing
head -5 suspicious_data.csv | faery input stdin --dimensions 640x480 \
    output file test_output.es 2>&1 | grep -q "Error" && \
    echo "Invalid format detected" || echo "Format OK"
```

### Robust processing

```sh
#!/bin/bash
# Robust stdin processing script

set -e  # Exit on any error

# Check if stdin has data
if [ -t 0 ]; then
    echo "Error: No data on stdin" >&2
    exit 1
fi

# Process with error handling
faery input stdin --dimensions 640x480 output file output.es || {
    echo "Error: Failed to process stdin data" >&2
    exit 1
}

echo "Successfully processed stdin data"
```

## Performance considerations

### Large data streams

```sh
# Use buffering for large streams
stdbuf -o0 -e0 cat large_events.csv | \
    faery input stdin --dimensions 640x480 output file large_output.es

# Process in chunks for memory efficiency
split -l 1000000 huge_events.csv chunk_ && \
for chunk in chunk_*; do
    cat "$chunk" | faery input stdin --dimensions 640x480 \
        output file "${chunk}.es"
done
```

### Progress monitoring

```sh
# Monitor processing progress
pv large_events.csv | faery input stdin --dimensions 640x480 output file output.es

# With line counting
cat events.csv | pv -l | faery input stdin --dimensions 640x480 output file output.es
```

## Troubleshooting

### Common issues

**"No data available"**
```
ValueError: No data available on stdin
```
- Verify data is being piped to stdin: `echo "test" | your_command`
- Check for empty input files
- Ensure proper pipeline construction

**"Invalid CSV format"**
```
ValueError: CSV parsing error
```
- Verify column indices with `--csv-*-index` parameters
- Check separator with `--csv-separator`
- Validate data format: `head -5 data.csv`

**"Dimension mismatch"**
```
ValueError: Coordinates exceed dimensions
```
- Check that x/y values fit within specified dimensions
- Verify coordinate columns are correctly mapped
- Consider using larger dimensions parameter

### Debugging techniques

```sh
# Preview data format
head -5 data.csv

# Test with minimal data
echo "1000,100,200,1" | faery input stdin --dimensions 640x480 output file test.es

# Validate column mapping
awk -F, '{print "t:"$1" x:"$2" y:"$3" p:"$4}' data.csv | head -5

# Check for invalid characters
cat data.csv | tr -cd '[:print:]\n' | head -5
```

### Performance optimization

```sh
# Disable header processing if not needed
faery input stdin --dimensions 640x480 --no-csv-has-header output file output.es

# Use appropriate buffer sizes
stdbuf -i8192 -o8192 faery input stdin --dimensions 640x480 output file output.es

# Parallel processing for multiple files
find . -name "*.csv" | xargs -P4 -I{} sh -c 'cat {} | faery input stdin --dimensions 640x480 output file {}.es'
```

Standard input processing in Faery provides a powerful way to integrate neuromorphic data processing into existing command-line workflows and enables seamless interoperability with other tools and systems.
