# File inputs

Faery supports reading neuromorphic event data from multiple file formats. This section covers all supported file types and their specific handling requirements.

## AEDAT4 files (.aedat4)

AEDAT4 is the standard format from iniVation for event camera data. It's a container format that can hold multiple data streams.

### Command line
```sh
# Basic file conversion
faery input file input.aedat4 output file output.es

# Specify a specific track/stream (useful for multi-stream files)
faery input file input.aedat4 --track-id 1 output file output.es
```

### Python
```python
import faery

# Read from AEDAT4 file
stream = faery.events_stream_from_file("input.aedat4")

# Specify track ID for multi-stream files
stream = faery.events_stream_from_file("input.aedat4", track_id=1)
```

### Format details
- **Multi-stream support**: AEDAT4 files can contain multiple data streams. Use `track_id` to select a specific stream (defaults to the first event stream)
- **Metadata**: Contains sensor dimensions and other metadata
- **Compression**: Supports internal compression

## ES files (.es)

ES is Faery's native binary format, optimized for fast reading and minimal overhead.

### Command line
```sh
# Convert to ES format
faery input file input.raw output file output.es

# Read from ES with custom start time
faery input file input.es --t0 1000000 output file output.mp4
```

### Python
```python
import faery

# Read ES file
stream = faery.events_stream_from_file("input.es")

# Specify start time offset (in microseconds)
stream = faery.events_stream_from_file("input.es", t0=1000000 * faery.us)
```

### Format details
- **Native format**: Optimized for Faery's internal event representation
- **Time offset**: Use `t0` parameter to specify initial timestamp
- **Compression**: Supports LZ4 and ZSTD compression

## Prophesee RAW files (.raw)

Prophesee RAW is the native format from Prophesee event cameras.

### Command line
```sh
# Convert Prophesee RAW file
faery input file input.raw output file output.aedat4

# Specify dimensions if not in header
faery input file input.raw --dimensions-fallback 640x480 output file output.es

# Override version detection
faery input file input.raw --version-fallback evt3 output file output.es
```

### Python
```python
import faery

# Read Prophesee RAW file
stream = faery.events_stream_from_file("input.raw")

# Specify fallback dimensions and version
stream = faery.events_stream_from_file(
    "input.raw",
    dimensions_fallback=(640, 480),
    version_fallback="evt3"
)
```

### Format details
- **Header detection**: Faery automatically detects sensor dimensions and format version from the file header
- **Fallbacks**: Use `dimensions_fallback` and `version_fallback` when header information is missing
- **Versions**: Supports EVT2, EVT2.1, and EVT3 formats

## DAT files (.dat)

DAT is another format variant, similar to EVT but with different encoding.

### Command line
```sh
# Convert DAT file
faery input file input.dat output file output.aedat4

# Specify fallback parameters
faery input file input.dat --dimensions-fallback 1280x720 --version-fallback dat2 output file output.es
```

### Python
```python
import faery

# Read DAT file with fallbacks
stream = faery.events_stream_from_file(
    "input.dat",
    dimensions_fallback=(1280, 720),
    version_fallback="dat2"
)
```

### Format details
- **Versions**: Supports DAT1 and DAT2 formats
- **Similar to EVT**: Uses similar fallback mechanisms as EVT files

## CSV files (.csv)

For human-readable data or custom formats, Faery supports CSV files with configurable column mapping.

### Command line
```sh
# Read CSV with default column mapping (t,x,y,p)
faery input file events.csv output file output.es

# Custom column indices and separator
faery input file events.csv --csv-t-index 0 --csv-x-index 1 --csv-y-index 2 --csv-p-index 3 --csv-separator ";" output file output.es

# CSV without header
faery input file events.csv --no-csv-has-header output file output.es
```

### Python
```python
import faery

# Read CSV with default settings
stream = faery.events_stream_from_file("events.csv")

# Custom CSV properties
csv_props = faery.CsvProperties(
    has_header=True,
    separator=";",
    t_index=0,
    x_index=1,
    y_index=2,
    p_index=3
)
stream = faery.events_stream_from_file("events.csv", csv_properties=csv_props)
```

### Format details
- **Column mapping**: Configure which columns contain timestamp, x, y, and polarity data
- **Headers**: Optionally skip the first row if it contains column headers
- **Separators**: Support for different delimiter characters

## Format detection

Faery automatically detects file formats based on file extensions:

| Extension | Format |
|-----------|--------|
| `.aedat4` | AEDAT4 |
| `.es` | ES |
| `.raw` | Prophesee EVT |
| `.dat` | DAT |
| `.csv` | CSV |

### Override format detection

### Command line
```sh
# Force interpretation as specific format
faery input file mystery_file --file-type csv output file output.es
```

### Python
```python
# Override automatic detection
stream = faery.events_stream_from_file("mystery_file", file_type="csv")
```

## Common parameters

All file input methods support these common parameters:

- **`dimensions_fallback`**: Default sensor dimensions when not available in file header
- **`file_type`**: Override automatic format detection
- **`track_id`**: Select specific stream from multi-stream files (AEDAT4 only)

## Troubleshooting

**File not recognized**: Use `--file-type` or `file_type=` to override detection
**Dimension errors**: Specify `--dimensions-fallback` or `dimensions_fallback=`
**Multi-stream confusion**: Use `--track-id` or `track_id=` for AEDAT4 files
**CSV parsing issues**: Check column indices and separator settings
