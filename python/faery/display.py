import functools
import os
import sys
import typing

from . import events_stream_state, frame_stream_state, timestamp

ANSI_COLORS_ENABLED = os.getenv("ANSI_COLORS_DISABLED") is None


def generate_progress_bar(width: int, progress: typing.Optional[float]) -> str:
    """Generates a progress bar compatible with terminals.

    Args:
        width (int): The progress bar width in characters.
        progress (typing.Optional[tuple[float, float]]): None yields an indeterminate progress bar, a value in the range [0, 1] yields a progress bar.

    Returns:
        str: The progress bar as a string, without line breaks.
    """

    width = max(width, 3)
    if progress is None:
        return "|{}|".format("░" * (width - 2))
    progress = max(0.0, min(1.0, progress))
    progress_fill = round((width - 2) * progress)
    return "|{}{}|".format(
        "█" * progress_fill,
        "–" * (width - 2 - progress_fill),
    )


def progress_bar_implementation(
    state: typing.Union[
        events_stream_state.EventsStreamState,
        events_stream_state.FiniteEventsStreamState,
        events_stream_state.RegularEventsStreamState,
        events_stream_state.FiniteRegularEventsStreamState,
        frame_stream_state.FrameStreamState,
        frame_stream_state.FiniteFrameStreamState,
        frame_stream_state.RegularFrameStreamState,
        frame_stream_state.FiniteRegularFrameStreamState,
    ],
    clear_after_last: bool,
):
    if isinstance(
        state,
        (
            events_stream_state.EventsStreamState,
            events_stream_state.RegularEventsStreamState,
            frame_stream_state.FrameStreamState,
            frame_stream_state.RegularFrameStreamState,
        ),
    ):
        if isinstance(
            state,
            (
                events_stream_state.EventsStreamState,
                events_stream_state.RegularEventsStreamState,
            ),
        ):
            if state.packet == "start":
                suffix = ""
                last = False
            elif state.packet == "end":
                suffix = ""
                last = True
            else:
                suffix = state.packet.time_range[1].to_timecode()
                last = False
        else:
            if state.frame == "start":
                suffix = ""
                last = False
            elif state.frame == "end":
                suffix = ""
                last = True
            else:
                suffix = state.frame.t.to_timecode()
                last = False
        columns = os.get_terminal_size().columns
        progress_bar_width = columns - len(suffix) - 2
        if progress_bar_width > 2:
            sys.stdout.write(
                f"\r{generate_progress_bar(width=progress_bar_width, progress=None)} {suffix}"
            )
        else:
            sys.stdout.write(f"\r{' ' * columns}\r{suffix}")
        if last:
            if clear_after_last:
                sys.stdout.write(f"\r{' ' * columns}\r")
            else:
                sys.stdout.write("\n")
        sys.stdout.flush()
    elif isinstance(
        state,
        (
            events_stream_state.FiniteEventsStreamState,
            events_stream_state.FiniteRegularEventsStreamState,
            frame_stream_state.FiniteFrameStreamState,
            frame_stream_state.FiniteRegularFrameStreamState,
        ),
    ):
        if state.progress <= 0.0:
            prefix = "0.00 %"
        elif state.progress >= 1.0:
            prefix = " 100 %"
        else:
            progress = round(state.progress * 100.0, 2)
            if progress < 10.0:
                prefix = f"{progress:.2f} %"
            else:
                progress = round(state.progress * 100.0, 1)
                if progress < 100.0:
                    prefix = f"{progress:.1f} %"
                else:
                    prefix = " 100 %"
        if isinstance(
            state,
            (
                events_stream_state.FiniteEventsStreamState,
                events_stream_state.FiniteRegularEventsStreamState,
            ),
        ):
            if state.packet == "start":
                numerator = state.stream_time_range[0]
                last = False
            elif state.packet == "end":
                numerator = state.stream_time_range[1]
                last = True
            else:
                numerator = state.packet.time_range[1]
                last = False
        else:
            if state.frame == "start":
                numerator = state.stream_time_range[0]
                last = False
            elif state.frame == "end":
                numerator = state.stream_time_range[1]
                last = True
            else:
                numerator = state.frame.t
                last = False
        suffix = (
            f"{numerator.to_timecode()} / {state.stream_time_range[1].to_timecode()}"
        )
        columns = os.get_terminal_size().columns
        progress_bar_width = columns - len(prefix) - len(suffix) - 3
        if progress_bar_width > 2:
            sys.stdout.write(
                f"\r{prefix} {generate_progress_bar(width=progress_bar_width, progress=state.progress)} {suffix}"
            )
        else:
            sys.stdout.write(f"\r{' ' * columns}\r{prefix} {suffix}")
        if last:
            if clear_after_last:
                sys.stdout.write(f"\r{' ' * columns}\r")
            else:
                sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        raise Exception(f"unsupported state type {state}")


progress_bar = functools.partial(progress_bar_implementation, clear_after_last=False)
progress_bar_fold = functools.partial(
    progress_bar_implementation, clear_after_last=True
)


def format_bold(message: str) -> str:
    """Surrounds the message with ANSI escape characters for bold formatting.

    Args:
        message (str): A message to be displayed in a terminal.

    Returns:
        str: The message surrounded with ANSI escape characters, or the original message if the environment variable ``ANSI_COLORS_ENABLED`` is not set.
    """
    if ANSI_COLORS_ENABLED:
        return f"\033[1m{message}\033[0m"
    return message


def format_color(
    message: str,
    color: typing.Literal[
        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white"
    ],
) -> str:
    """Surrounds the message with ANSI escape characters for color formatting.

    Args:
        message (str): A message to be displayed in a terminal.

    Returns:
        str: The message surrounded with ANSI escape characters, or the original message if the environment variable ``ANSI_COLORS_ENABLED`` is not set.
    """
    if ANSI_COLORS_ENABLED:
        if color == "black":
            color_code = 0
        elif color == "red":
            color_code = 1
        elif color == "green":
            color_code = 2
        elif color == "yellow":
            color_code = 3
        elif color == "blue":
            color_code = 4
        elif color == "magenta":
            color_code = 5
        elif color == "cyan":
            color_code = 6
        elif color == "white":
            color_code = 7
        else:
            raise Exception(f'unknown color "{color}"')
        return f"\033[3{color_code}m{message}\033[0m"
    return message
