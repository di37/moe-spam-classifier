"""Movement shared by the videos: a token travelling along a path, and tokens streaming in.

A path is a list of waypoints ``(x, y, piece)``: a position, and the part of the drawing the
token is in when it gets there.
"""

import numpy as np
from matplotlib.patches import Circle

from moe_spam.viz.style import INK, SURFACE
from moe_spam.viz.video import ease

SLOW_TOKENS, SLOW_SHARE = 4, 0.5  # by default the first 4 tokens take half the stream


def path_lengths(waypoints):
    """Returns the distance along ``waypoints`` to each of them."""
    points = np.array([(x, y) for x, y, _ in waypoints])
    return np.r_[0, np.cumsum(np.hypot(*np.diff(points, axis=0).T))]


def progress(waypoints, index):
    """Returns how far along ``waypoints`` (0 to 1, by length) waypoint ``index`` is."""
    lengths = path_lengths(waypoints)
    return float(lengths[index] / lengths[-1])


def locate(waypoints, t):
    """Returns ``(x, y, piece)`` of a token ``t`` (0 to 1, by length) along ``waypoints``.

    The piece is the one the token is heading for (or has reached).
    """
    points, lengths = np.array([(x, y) for x, y, _ in waypoints]), path_lengths(waypoints)
    at = float(np.clip(t, 0, 1)) * lengths[-1]
    k = min(int(np.searchsorted(lengths, at, side="right")) - 1, len(points) - 2)
    share = (at - lengths[k]) / max(lengths[k + 1] - lengths[k], 1e-9)
    x, y = points[k] + share * (points[k + 1] - points[k])
    return float(x), float(y), waypoints[k + 1][2] if share > 0 else waypoints[k][2]


def locate_in_steps(waypoints, t):
    """Like ``locate``, but every step between two waypoints takes the same time.

    Within a step the token eases out of one waypoint and into the next, so it slows down
    at every stop. Waypoint ``k`` of ``n`` is reached at ``t = k / (n - 1)``.

    Returns:
        ``(x, y, piece)``: the piece is the one heading for (or reached).
    """
    steps = len(waypoints) - 1
    at = float(np.clip(t, 0, 1)) * steps
    k = min(int(at), steps - 1)
    share = ease(at - k)
    (x0, y0, here), (x1, y1, there) = waypoints[k], waypoints[k + 1]
    return x0 + share * (x1 - x0), y0 + share * (y1 - y0), there if share > 0 else here


def draw_pulse(ax, x, y, label, radius=0.75, fontsize=10, lift=1.3, side="above"):
    """Draws the travelling token at ``(x, y)``, with its ``label`` above it or on its left.

    Args:
        ax: The axes to draw on.
        x: Where the token is.
        y: Where the token is.
        label: The token's name.
        radius: The dot's radius.
        fontsize: The label's size.
        lift: How far above the dot the label sits (``side="above"``).
        side: ``"above"`` or ``"left"``.
    """
    ax.add_patch(Circle((x, y), radius, facecolor=INK, edgecolor=SURFACE, lw=1.5, zorder=6))
    if side == "left":
        ax.text(x - radius - 0.6, y, label, ha="right", va="center", fontsize=fontsize,
                color=INK, weight="bold", zorder=6)
    else:
        ax.text(x, y + lift, label, ha="center", fontsize=fontsize, color=INK, weight="bold",
                zorder=6)


def token_progress(t, n, slow_tokens=SLOW_TOKENS, slow_share=SLOW_SHARE):
    """Returns how many of ``n`` tokens have streamed in at progress ``t``, as a float.

    The first ``slow_tokens`` take ``slow_share`` of the stream, so they can be followed;
    the rest go faster.
    """
    slow = min(slow_tokens, n)
    if t <= slow_share:
        return slow * t / slow_share
    return min(n, slow + (n - slow) * (t - slow_share) / (1 - slow_share))
