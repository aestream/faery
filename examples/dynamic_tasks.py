import pathlib

import faery


def create_kinectograph(colormap: faery.Colormap) -> faery.Task:

    @faery.task(suffix=".png", icon="🎨", name=f"kinectograph_{colormap.name}")
    def kinectograph(
        input: pathlib.Path,
        output: pathlib.Path,
        start: faery.TimeOrTimecode,
        end: faery.TimeOrTimecode,
    ):
        (
            faery.events_stream_from_file(input)
            .time_slice(start=start, end=end)
            .to_kinectograph(on_progress=faery.progress_bar_fold)
            .scale()
            .render(color_theme=faery.LIGHT_COLOR_THEME.replace(colormap=colormap))
            .to_file(path=output)
        )

    return kinectograph


# Generate tasks with decorators dynamically.
cyclic_colormaps_tasks: list[faery.Task] = []
for colormap in faery.colormaps_list():
    if colormap.type == "cyclic":
        # The decorated `kinectograph` task captures the `colormap` variable but evaluates it later
        # (when job_manager executes the task as part of its "run" function).
        # If we declared the task below instead of using a wrapper function (`create_kinectograph`),
        # the variable `colormap` would always have the value `faery.colormaps.vik_o`.
        # See https://docs.python-guide.org/writing/gotchas/#late-binding-closures for details.
        cyclic_colormaps_tasks.append(create_kinectograph(colormap))


# Generate tasks with classes dynamically.
# This is equivalent to the decorator method.
# Classes are more verbose but they can be more convenient in some cases.
# Internally, decorators declare classes.
sequential_colormaps_tasks: list[faery.Task] = []
for colormap in faery.colormaps_list():
    if colormap.type == "sequential":

        class Kinectograph(faery.Task):

            def __init__(self, colormap: faery.Colormap):
                self.colormap = colormap

            def suffix(self) -> str:
                return ".png"

            def icon(self) -> str:
                return "🎨"

            def name(self) -> str:
                return f"kinectograph_{self.colormap.name}"

            def __call__(
                self,
                input: pathlib.Path,
                output: pathlib.Path,
                start: faery.Time,
                end: faery.Time,
            ):
                (
                    faery.events_stream_from_file(input)
                    .time_slice(start=start, end=end)
                    .to_kinectograph(on_progress=faery.progress_bar_fold)
                    .scale()
                    .render(
                        color_theme=faery.LIGHT_COLOR_THEME.replace(
                            colormap=self.colormap
                        )
                    )
                    .to_file(path=output)
                )

        # We avoid the late binding issue here since we pass `colormap` to Kinectograph's `__init__` function.
        sequential_colormaps_tasks.append(Kinectograph(colormap=colormap))


job_manager = faery.JobManager(
    output_directory=faery.dirname.parent / "tests" / "data_generated" / "renders"
)

job_manager.add(
    faery.dirname.parent / "tests" / "data" / "dvs.es",
    "00:00:00.000000",
    "00:00:00.999001",
    cyclic_colormaps_tasks + sequential_colormaps_tasks,
)

job_manager.run()
