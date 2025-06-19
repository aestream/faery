# Usage: Batch processing

This page describes how Faery can be used to render and analyze many recordings at once.

## Setup

Ensure you have [installed Faery](#installation) and have access to the `faery` command in your terminal.

## Workflow

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
