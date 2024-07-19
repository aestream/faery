from dataclasses import dataclass

from typing import Any

@dataclass
class CliConfig:
    input: Any = None
    output: Any = None
