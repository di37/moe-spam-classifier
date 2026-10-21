"""Movement shared by the videos: a token travelling along a path, and tokens streaming in.

A path is a list of waypoints ``(x, y, piece)``: a position, and the part of the drawing the
token is in when it gets there.
"""

import numpy as np
from matplotlib.patches import Circle

from moe_spam.viz.style import INK, SURFACE

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


def draw_pulse(ax, x, y, label, radius=0.75, fontsize=10, lift=1.3):
    """Draws the travelling token at ``(x, y)``, with its ``label`` ``lift`` above it."""
    ax.add_patch(Circle((x, y), radius, facecolor=INK, edgecolor=SURFACE, lw=1.5, zorder=6))
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
