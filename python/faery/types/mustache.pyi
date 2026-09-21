class Job:
    input: str
    start: str
    end: str
    nickname: str | None

    def __init__(self, input: str, start: str, end: str, nickname: str | None): ...

def render(template: str, jobs: list[Job]) -> str: ...
