"""The look shared by every frame: colors, fonts, figure size and matplotlib style.

The expert colors are the first three slots of the dataviz reference palette, which stay
distinguishable together under color-vision deficiencies. Everything that is not an expert
is drawn in ink.
"""

from matplotlib.colors import to_rgb

EXPERT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
HIGHLIGHT = "#f0efec"  # neutral fill for whatever is being shown
LAYER_COLORS = {"dense": MUTED, "wide": INK_2, "moe": INK}  # one FFN, one 3× wider, MoE
ERROR = "#d03b3b"      # status "critical", used only for misclassified messages

MONO = "DejaVu Sans Mono"  # ships with matplotlib, so the layout is the same everywhere
CHAR_EM = 0.602            # advance width of one DejaVu Sans Mono character, in ems
LINE_EM = 2.0              # height of one line of token boxes, in ems
MATH_FONT = "cm"           # equations ($...$) in Computer Modern, LaTeX's own font

DPI = 100
FIGURE_PX = (1600, 900)
STYLE = {
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 11,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
}


def ink_on(fill):
    """Returns ink or white: whichever text color reads better on ``fill``."""
    return INK if _contrast(fill, INK) >= _contrast(fill, "white") else "white"


def blend(color, amount):
    """Returns ``amount`` of ``color`` over the surface, as RGB (0 = surface, 1 = color)."""
    return tuple(amount * c + (1 - amount) * s
                 for c, s in zip(to_rgb(color), to_rgb(SURFACE), strict=True))


def _luminance(color):
    """WCAG relative luminance."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(color)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b):
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)
