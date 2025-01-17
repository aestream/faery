from . import colormaps
from .colormaps._base import Color as Color
from .colormaps._base import Colormap as Colormap
from .colormaps._base import ColorTheme as ColorTheme
from .colormaps._base import color_to_floats as color_to_floats
from .colormaps._base import color_to_hex_string as color_to_hex_string
from .colormaps._base import color_to_ints as color_to_ints
from .colormaps._base import gradient as gradient

LIGHT_COLOR_THEME: ColorTheme = ColorTheme(
    background="#FFFFFFFF",
    labels="#000000FF",
    axes="#000000FF",
    grid="#000000FF",
    subgrid="#00000050",
    lines=["#CCCCCCFF", "#000000FF"],
    colormap=colormaps.batlow,
)

DARK_COLOR_THEME: ColorTheme = ColorTheme(
    background="#191919FF",
    labels="#FFFFFFFF",
    axes="#FFFFFFFF",
    grid="#FFFFFFFF",
    subgrid="#FFFFFF50",
    lines=["#142C54FF", "#4285F4FF"],
    colormap=colormaps.batlow,
)
