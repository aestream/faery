import subprocess
import sys

from . import command


class Command(command.Command):
    def usage(self) -> tuple[list[str], str]:
        return (["faery run"], "run a Faery script")

    def first_block_keywords(self) -> set[str]:
        return {"run"}

    def run(self, arguments: list[str]):
        parser = self.parser()
        parser.add_argument(
            "--input",
            "-i",
            default="faery_script.py",
            help="input script to run",
        )
        parser.add_argument("rest", nargs="*")
        args = parser.parse_args(args=arguments)
        process = subprocess.run([sys.executable, args.input] + args.rest, check=False)
        sys.exit(process.returncode)
