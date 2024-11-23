import pathlib
import typing

class Task:
    task_hash: str
    task_code: str

    def __init__(self, task_hash: str, task_code: str): ...

def read(path: typing.Union[pathlib.Path, str]) -> dict[str, Task]: ...
def write(job_metadata: dict[str, Task], path: typing.Union[pathlib.Path, str]): ...
