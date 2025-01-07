![faery logo](faery_logo.png)

Faery converts neuromorphic event-based data between formats. It can also generate videos, spectrograms, and event rate curves.

- [Using Faery from the command line](#using-faery-from-the-command-line)
  - [Setup](#setup)
  - [Examples](#examples)
- [Using Faery in a Python script](#using-faery-in-a-python-script)
  - [Setup](#setup-1)
  - [Examples](#examples-1)
- [Using Faery to render and analyze many recordings](#using-faery-to-render-and-analyze-many-recordings)
  - [Setup](#setup-2)
  - [Workflow](#workflow)
- [Local development](#local-development)
  - [Setup the environment](#setup-the-environment)
  - [Format and lint](#format-and-lint)
  - [Test](#test)
  - [Upload a new version](#upload-a-new-version)
- [Acknowledgements](#acknowledgements)

## Using Faery from the command line

### Setup

1. Install pipx (https://pipx.pypa.io/stable/installation/)

2. Install Faery by running `pipx install faery`

3. Run `faery --help` to see a list of options, `faery list-filters` to list available filters, and `faery filter <filter> --help` for help on a specific filter

> Note: you can use a virtual environment instead of pipx, see [Using Faery in a Python script](#using-faery-in-a-python-script) for instructions.

### Examples

```sh
# Convert a Prophesee raw file (input.raw) to AEDAT (output.aedat)
faery input file input.raw output file output.aedat

# Render an event file (input.es) as a real-time video (output.mp4)
faery input file input.es filter regularize 60.0 filter render exponential 0.2 starry_night output file output.mp4

# Render an event file (input.es) as a video 10 x slower than real-time (output.mp4)
# The second render parameter (0.03) is the exponential decay constant.
# Slow-motion videos look better with shorter decays but it does not need to be scaled like regularize,
# which controls the playback speed.
faery input file input.es filter regularize 600.0 filter render exponential 0.03 starry_night output file output.mp4

# Render an event file (input.es) as frames (frames/*.png)
faery input file input.es filter regularize 60.0 filter render exponential 0.2 starry_night output files 'frames/{index:04}.png'

# Print ON events to the terminal
faery input file input.aedat filter remove-off-events

# Read event data from UDP and write it to a CSV file (output.csv)
faery input udp 0.0.0.0:3000 output file output.csv
```

## Using Faery in a Python script

### Setup

```sh
python3 -m venv .venv
source .venv/bin/activate # 'venv\Scripts\activate' on Windows
pip install faery
```

### Examples

See _examples_ in this repository.

## Using Faery to render and analyze many recordings

### Setup

1. Install pipx (https://pipx.pypa.io/stable/installation/)

2. Install Faery by running `pipx install faery`

> Note: you can use a virtual environment instead of pipx, see [Using Faery in a Python script](#using-faery-in-a-python-script) for instructions.

### Workflow

1. Create a directory called _my-wonderful-project_ (or any other name), a subdirectory called _recordings_ (this must be called _recordings_), and move the files to analyze to _recordings_ (the file names do not matter). Faery supports AEDAT, RAW, DAT, ES, and CSV files.

    ```txt
    my-wonderful-project
    └── recordings
        ├── file_1.raw
        ├── file_2.raw
        ├── ...
        └── file_n.raw
    ```

2. Generate a render and analysis script with Faery.

    ```sh
    cd path/to/my-wonderful-project
    faery init
    ```

    Faery will read _recordings_, calculate the time range of each recording, and create the file _faery_script.py_. This can take a little while if there are many recordings or if they are large.

    You can also use `faery init --generate-nicknames` to use easy-to-remember nicknames for the files instead of their original names.

    If you use Visual Studio Code and pipx, consider using `faery init --vscode`. This will create a VS code settings file in _my-wonderful-project_ and enable type completion in _faery_script.py_.

3. Run the script.

    ```sh
    faery run
    ```

    Faery will execute the script, which generates assets in the _renders_ directory.

    > Note: if you installed Faery in a virtual environment, you can also run the script directly with `python faery_script.py`

4. Edit the script.

    You can modify _faery_script.py_ to analyze shorter time slices, use different colormaps, or generate slow motion videos. After editing the script, run it again with `faery run`.

    Faery keeps track of completed renders, hence you do not need to delete past jobs before running the script again. For instance, if you run the default script once and find a time window of interest for which you wish to generate a slow motion video, we recommend proceeding as follows.

    a. Add a slow motion video task to the script after `real_time_video`.

    ```py
    @faery.task(suffix=".mp4", icon="🎬")
    def real_time_video(...):
        ...

    @faery.task(suffix=".mp4", icon="🎬")
    def slow_motion_video(
        input: pathlib.Path,
        output: pathlib.Path,
        start: faery.Time,
        end: faery.Time,
    ):
        (
            faery.events_stream_from_file(input)
            .time_slice(start=start, end=end)
            .regularize(frequency_hz=6000.0) # 100 x slower than real time
            .render(
                decay="exponential",
                tau="00:00:00.002000", # faster decay
                colormap=faery.colormaps.starry_night,
            )
            .scale()
            .to_file(output, on_progress=faery.progress_bar_fold)
        )
    ```

    b. Add a new job at the end of the file, before `job_manager.run()`. You can use the same nickname as long as the time range is different.

    ```py
    tasks: list[faery.Task] = [kinectograph, kinectograph_dense, real_time_video]

    # original job (stays in the script)
    job_manager.add(
        faery.dirname / "recordings" / "dvs.es",
        "00:00:00.000000",
        "00:00:00.999001",
        tasks,
        nickname="optimistic-tarsier"
    )

    # new job (added to the script)
    job_manager.add(
        faery.dirname / "recordings" / "dvs.es",
        "00:00:00.100000",
        "00:00:00.200000",
        [kinectograph, kinectograph_dense, slow_motion_video],
        nickname="optimistic-tarsier"
    )

    job_manager.run()
    ```

    Since we use `slow_motion_video` (100 x slower than real-time) on a 100 ms slice, the resulting video will be 10 seconds long.

## Local development

### Setup the environment

Local build (first run).

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh # see https://rustup.rs
python3 -m venv .venv
source .venv/bin/activate
# x86 platforms may need to install https://www.nasm.us
pip install --upgrade pip
pip install maturin
maturin develop  # or maturin develop --release to build with optimizations
```

Local build (subsequent runs).

```sh
source .venv/bin/activate
maturin develop  # or maturin develop --release to build with optimizations
```

### Format and lint

```sh
cargo fmt
cargo clippy
pip install isort black pyright
isort .; black .; pyright .
```

### Test

```sh
pip install pytest
pytest tests
```

### Upload a new version

1. Update the version in _pyproject.toml_.

2. Push the changes

3. Create a new release on GitHub. GitHub actions should build wheels and push them to PyPI.

## Acknowledgements

Faery was initiated at the [2024 Telluride neuromorphic workshop](https://sites.google.com/view/telluride-2024/) by

-   [Alexandre Marcireau](https://github.com/amarcireau)
-   [Jens Egholm Pedersen](https://github.com/jegp)
-   [Gregor Lenz](https://github.com/biphasic)
-   [Gregory Cohen](https://github.com/gcohen)

License: LGPLv3.0
