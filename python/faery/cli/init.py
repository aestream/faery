import importlib.resources
import json
import pathlib
import sys
import typing

import faery

from . import command, coolname


class Command(command.Command):
    @typing.override
    def usage(self) -> tuple[list[str], str]:
        return (["faery init"], "initialize a Faery script")

    @typing.override
    def first_block_keywords(self) -> set[str]:
        return {"init"}

    @typing.override
    def run(self, arguments: list[str]):
        parser = self.parser()
        parser.add_argument(
            "--input",
            "-i",
            nargs="*",
            default=["recordings/*", "ignore:recordings/.*"],
            help="select input files that match the given glob pattern (default: --input 'recordings/*' --input 'ignore:recordings/.*'), if the name starts with \"ignore:\" (for instance \"ignore:recordings/*.png\"), matching files are removed from the selection",
        )
        parser.add_argument(
            "--output",
            "-o",
            default="faery_script.py",
            help="path of the output script (default: faery_script.py)",
        )
        parser.add_argument(
            "--generate-nicknames",
            "-g",
            action="store_true",
            help="generate cool nicknames for the recordings",
        )
        parser.add_argument(
            "--force",
            "-f",
            action="store_true",
            help="replace the script file if it already exists",
        )
        parser.add_argument(
            "--vscode",
            "-c",
            action="store_true",
            help="create .vscode/settings and define python.defaultInterpreterPath",
        )
        parser.add_argument("--template", "-t", help="path of the script template")
        parser.add_argument(
            "--export-template",
            "-e",
            help="path where to *write* the default template",
        )
        args = parser.parse_args(args=arguments)
        if args.export_template is not None:
            if (
                len(args.input) != 1
                or args.input[0] != "recordings/*"
                or args.output != "faery_script.py"
                or args.template is not None
            ):
                sys.stderr.write(f"--export-template cannot appear with other flags\n")
                sys.exit(1)
            with (
                importlib.resources.files(faery)
                .joinpath("cli/faery_script.mustache")
                .open("r") as template_file
            ):
                template = template_file.read()
            with open(args.export_template, "w") as output:
                output.write(template)
            sys.exit(0)
        if args.template is None:
            with (
                importlib.resources.files(faery)
                .joinpath("cli/faery_script.mustache")
                .open("r") as template_file
            ):
                template = template_file.read()
        else:
            with open(args.template, "r") as template_file:
                template = template_file.read()
        output = pathlib.Path(args.output).resolve()
        if not args.force and output.is_file():
            sys.stderr.write(f'"{output}" already exists\n')
            sys.exit(1)
        resolved_to_original: dict[pathlib.Path, pathlib.Path] = {}
        pattern_and_count: list[tuple[str, int]] = []
        for pattern in args.input:
            count = 0
            if pattern.startswith("ignore:"):
                for path in pathlib.Path().glob(pattern[len("ignore:") :]):
                    resolved = path.resolve()
                    if resolved in resolved_to_original:
                        del resolved_to_original[resolved]
                        count += 1
                pattern_and_count.append((pattern, count))
            else:
                for path in pathlib.Path().glob(pattern):
                    resolved_to_original[path.resolve()] = path
                    count += 1
                pattern_and_count.append((pattern, count))
        jobs = []
        contents = faery.mustache.render(template=template, jobs=jobs)
        with open(args.output, "w") as output_file:
            output_file.write(contents)

        if args.vscode:
            vscode_directory = pathlib.Path(args.output).resolve().parent / ".vscode"
            vscode_directory.mkdir(exist_ok=True)
            if (vscode_directory / "settings.json").is_file():
                with open("settings.json") as settings_file:
                    settings = json.load(settings_file)
                if not "python.defaultInterpreterPath" in settings:
                    settings["python.defaultInterpreterPath"] = sys.executable
                    with open("settings.json", "w") as settings_file:
                        json.dump(settings, settings_file, indent=4)
            else:
                with open("settings.json", "w") as settings_file:
                    json.dump(
                        {"python.defaultInterpreterPath": sys.executable},
                        settings_file,
                        indent=4,
                    )

        if len(resolved_to_original) > 0:
            print("Read the time range of input files")
        if args.generate_nicknames:
            generated_nicknames = coolname.generate_distinct(len(resolved_to_original))
        else:
            generated_nicknames = None
        for index, (resolved_path, path) in enumerate(resolved_to_original.items()):
            if args.generate_nicknames:
                assert generated_nicknames is not None
                nickname = generated_nicknames[index]
                display_name = nickname
            else:
                nickname = None
                display_name = path.stem
            print(
                f"({index + 1}/{len(resolved_to_original)}) {faery.format_bold(display_name)} ({resolved_path})"
            )
            time_range = faery.events_stream_from_file(path=path).time_range()
            if path.is_absolute():
                input = f"{path}"
            else:
                parts_representation = " / ".join(f'"{part}"' for part in path.parts)
                input = f"faery.dirname / {parts_representation}"
            jobs.append(
                faery.mustache.Job(
                    input=input,
                    start=time_range[0].to_timecode(),
                    end=time_range[1].to_timecode(),
                    nickname=nickname,
                )
            )
            contents = faery.mustache.render(template=template, jobs=jobs)
            with open(args.output, "w") as output_file:
                output_file.write(contents)
        if len(jobs) == 0 and len(pattern_and_count) > 0:
            if len(pattern_and_count) == 1:
                sys.stderr.write(
                    f"No files matched the input pattern ({pattern_and_count[0]})\n"
                )
            else:
                sys.stderr.write(f"No files matched the input patterns:\n")
                for pattern, count in pattern_and_count:
                    if count == 0:
                        quantifier = "no files"
                    elif count == 1:
                        quantifier = "1 file"
                    else:
                        quantifier = f"{count} files"
                    if pattern.startswith("ignore:"):
                        sys.stderr.write(f'    - "{pattern}" removed {quantifier}\n')
                    else:
                        sys.stderr.write(f'    + "{pattern}" added {quantifier}\n')
