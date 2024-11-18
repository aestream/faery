import typing

class Job:
    input: str
    start: str
    end: str
    nickname: typing.Optional[str]

    def __init__(
        self, input: str, start: str, end: str, nickname: typing.Optional[str]
    ): ...

def render(template: str, jobs: list[Job]) -> str: ...
