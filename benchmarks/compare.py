"""Compare two saved benchmark runs and print a Markdown table.

    python3 benchmarks/compare.py                    # latest run vs the one before
    python3 benchmarks/compare.py baseline linear    # by label (latest match)
    python3 benchmarks/compare.py 0001 0003          # by run number

Values are median ms per packet. Change is relative to the first run; negative
is faster. Changes inside both runs' interquartile ranges are marked "~".
"""

import json
import pathlib
import sys

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def find_runs() -> list[pathlib.Path]:
    return sorted(RESULTS.glob("*/*.json"), key=lambda path: path.name)


def resolve(key: str, runs: list[pathlib.Path]) -> pathlib.Path:
    matches = [
        run for run in runs if run.name.startswith(key) or run.stem.endswith(f"_{key}")
    ]
    if not matches:
        sys.exit(f'no saved run matches "{key}" in {RESULTS}')
    return matches[-1]


def load(path: pathlib.Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text())
    rows = {}
    for bench in data["benchmarks"]:
        packets = bench["extra_info"].get("packets", 1)
        stats = bench["stats"]
        rows[bench["fullname"].split("::", 1)[1]] = {
            "group": bench["group"],
            "median": stats["median"] * 1e3 / packets,
            "q1": stats["q1"] * 1e3 / packets,
            "q3": stats["q3"] * 1e3 / packets,
        }
    return data["commit_info"], rows


def describe(path: pathlib.Path, commit: dict) -> str:
    dirty = "+dirty" if commit.get("dirty") else ""
    return f"`{path.stem}` ({commit.get('id', '?')[:8]}{dirty})"


def main() -> None:
    runs = find_runs()
    if len(sys.argv) == 3:
        before, after = resolve(sys.argv[1], runs), resolve(sys.argv[2], runs)
    elif len(runs) >= 2:
        before, after = runs[-2], runs[-1]
    else:
        sys.exit("need two saved runs (see benchmarks/run.sh)")
    commit_before, rows_before = load(before)
    commit_after, rows_after = load(after)

    print(f"{describe(before, commit_before)} → {describe(after, commit_after)}\n")
    print("| benchmark | before (ms/packet) | after (ms/packet) | change |")
    print("|---|---:|---:|---:|")
    names = sorted(
        set(rows_before) | set(rows_after),
        key=lambda name: ((rows_after.get(name) or rows_before[name])["group"], name),
    )
    for name in names:
        old, new = rows_before.get(name), rows_after.get(name)
        if old is None or new is None:
            value = new or old
            cells = ("—", f"{value['median']:.4f}") if old is None else (
                f"{value['median']:.4f}",
                "—",
            )
            print(f"| {name} | {cells[0]} | {cells[1]} | {'new' if old is None else 'removed'} |")
            continue
        change = (new["median"] - old["median"]) / old["median"] * 100
        overlap = new["q1"] <= old["q3"] and old["q1"] <= new["q3"]
        marker = "~" if overlap else ""
        print(
            f"| {name} | {old['median']:.4f} | {new['median']:.4f} | "
            f"{marker}{change:+.1f}% |"
        )


if __name__ == "__main__":
    main()
