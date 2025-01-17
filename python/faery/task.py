import ast
import dataclasses
import functools
import hashlib
import inspect
import pathlib
import sys
import textwrap
import typing

from . import display, timestamp

if typing.TYPE_CHECKING:
    from .types import job_metadata  # type: ignore
else:
    from .extension import job_metadata

dirname: pathlib.Path = pathlib.Path(inspect.stack()[-1].filename).parent
"""
Path of the main script's parent directory.

This is useful to define paths relative to the script rather than the current working directory.
"""


def hash(function: typing.Callable) -> str:
    function_string = textwrap.dedent(inspect.getsource(function))
    function_ast = ast.parse(function_string)
    function_ast_string = ast.dump(function_ast)
    return hashlib.sha3_224(function_ast_string.encode()).hexdigest()


def hash_file(path: pathlib.Path) -> str:
    hash_object = hashlib.sha3_224()
    with open(path, "rb") as file:
        while True:
            buffer = file.read(65536)
            if len(buffer) == 0:
                break
            hash_object.update(buffer)
    return hash_object.hexdigest()


class Task:
    def suffix(self) -> str:
        raise NotImplementedError()

    def icon(self) -> str:
        raise NotImplementedError()

    def name(self) -> str:
        raise NotImplementedError()

    def code(self) -> str:
        return textwrap.dedent(inspect.getsource(self.__call__))

    @functools.cached_property
    def hash(self) -> str:
        return hash(function=self.__call__)

    def __call__(
        self,
        input: pathlib.Path,
        output: pathlib.Path,
        start: timestamp.TimeOrTimecode,
        end: timestamp.TimeOrTimecode,
    ):
        raise NotImplementedError()


def task(suffix: str, icon: str = "", name: typing.Optional[str] = None):
    def task_generator(
        function: typing.Callable[
            [
                pathlib.Path,
                pathlib.Path,
                timestamp.TimeOrTimecode,
                timestamp.TimeOrTimecode,
            ],
            None,
        ]
    ) -> Task:
        class DecoratedTask(Task):

            @typing.override
            def suffix(self) -> str:
                return suffix

            @typing.override
            def icon(self) -> str:
                return icon

            @typing.override
            def name(self) -> str:
                return function.__name__.replace("_", "-") if name is None else name

            @typing.override
            def code(self) -> str:
                return textwrap.dedent(inspect.getsource(function))

            @functools.cached_property
            @typing.override
            def hash(self) -> str:
                return hash(function=function)

            @typing.override
            def __call__(
                self,
                input: pathlib.Path,
                output: pathlib.Path,
                start: timestamp.TimeOrTimecode,
                end: timestamp.TimeOrTimecode,
            ):
                return function(input, output, start, end)

        return DecoratedTask()

    return task_generator


@dataclasses.dataclass
class Job:
    input: pathlib.Path
    start: timestamp.TimeOrTimecode
    end: timestamp.TimeOrTimecode
    tasks: list[Task]
    name: str


class JobManager:
    def __init__(self, output_directory: pathlib.Path = dirname / "renders"):
        self.jobs: list[Job] = []
        self.triplets: set[tuple[str, str, str]] = set()
        self.output_directory = output_directory
        output_directory.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        input: pathlib.Path,
        start: timestamp.TimeOrTimecode,
        end: timestamp.TimeOrTimecode,
        tasks: list[Task],
        nickname: typing.Optional[str] = None,
    ):
        name = input.stem if nickname is None else nickname
        triplet = (
            name,
            timestamp.parse_time(start).to_timecode(),
            timestamp.parse_time(end).to_timecode(),
        )
        if triplet in self.triplets:
            raise Exception(
                f"Two jobs have the same name ({name}), start ({start}), and end ({end})"
            )
        self.triplets.add(triplet)
        tasks_names: set[str] = set()
        for task in tasks:
            task_name = f"{task.name()}{task.suffix()}"
            if name in tasks_names:
                raise Exception(
                    f"Two tasks have the same name + suffix ({task_name}) in the job ({name}, {start}, {end})"
                )
            tasks_names.add(task_name)
        self.jobs.append(
            Job(
                input=input,
                start=start,
                end=end,
                tasks=tasks,
                name=name,
            )
        )

    def run(self, force: bool = False):
        for index, job in enumerate(self.jobs):
            if index > 0:
                sys.stdout.write("\n")
            sys.stdout.write(
                f"({index + 1}/{len(self.jobs)}) {display.format_bold(job.name)} ({job.input})\n"
            )
            output_name = "_".join(
                (
                    job.name,
                    timestamp.parse_time(job.start).to_timecode_with_dashes(),
                    timestamp.parse_time(job.end).to_timecode_with_dashes(),
                )
            )
            output_directory = self.output_directory / output_name
            output_directory.mkdir(exist_ok=True)
            for task in job.tasks:
                output = (
                    self.output_directory
                    / output_name
                    / f"{output_name}_{task.name()}{task.suffix()}"
                )
                try:
                    metadata = job_metadata.read(output.parent / "metadata.toml")
                except:
                    metadata = {}
                task_name = task.name()
                skip = (
                    not force
                    and task_name in metadata
                    and metadata[task_name].task_hash == task.hash
                )
                sys.stdout.write(
                    f"{task.icon()} {display.format_bold(task.name())} ({output.name}{', skipped' if skip else ''})\n"
                )
                if not skip:
                    task(input=job.input, output=output, start=job.start, end=job.end)
                metadata[task_name] = job_metadata.Task(
                    task_hash=task.hash,
                    task_code=task.code(),
                )
                job_metadata.write(metadata, output.parent / "metadata.toml")
