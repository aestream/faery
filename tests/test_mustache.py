import importlib.resources
import random
import uuid

import faery


def test_mustache_template():
    with (
        importlib.resources.files(faery)
        .joinpath("cli/faery_script.mustache")
        .open("r") as template_file
    ):
        template = template_file.read()

    jobs: list[faery.mustache.Job] = []
    for _ in range(0, 3):
        start = 0
        end = 0
        while start >= end:
            start = random.randint(0, 1 << 32)
            end = random.randint(0, 1 << 32)
        jobs.append(
            faery.mustache.Job(
                input=str(uuid.uuid4()),
                start=faery.Time(microseconds=start).to_timecode(),
                end=faery.Time(microseconds=end).to_timecode(),
                nickname=str(uuid.uuid4()),
            )
        )
    contents = faery.mustache.render(template=template, jobs=jobs)
    for job in jobs:
        assert job.input in contents
        assert not f'"{job.input}"' in contents
        assert f'"{job.start}"' in contents
        assert f'"{job.end}"' in contents
        assert f'"{job.nickname}"' in contents
