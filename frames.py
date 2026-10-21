"""
Training frames: one JPEG per kept step, in a layout that never moves, so the video shows
only what the model changed.

  TokenRow          one example message as drawn: its tokens and the expert that ran each one
  FrameState        what the model does on the fixed test set at one step
  TrainingRecorder  keeps the loss / error / expert-load history and what the model did at
                    each kept step, then draws every frame at the end. Drawing last lets the
                    whole video share one zoom on the routing plot, chosen to fit the run.
"""
import glob
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # render off-screen; frames are written to disk
import matplotlib.pyplot as plt
import numpy as np

from panels import (
    AXIS,
    GRID,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    draw_load,
    draw_shares,
    draw_ternary,
    draw_tokens,
    ternary_view,
)

DPI = 100
FIGURE_PX = (1600, 900)
LOSS_LIMITS = (1e-3, 2.0)
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


@dataclass(frozen=True)
class TokenRow:
    label: str     # "ham" or "spam", the true label
    words: tuple   # what to print for each token, [CLS] first
    experts: tuple  # the expert that processed each token (0-based)
    p_spam: float  # the model's spam probability for the whole message


@dataclass(frozen=True)
class FrameState:
    cls_probs: np.ndarray     # [n, 3] layer-2 router probabilities of each test [CLS] token
    is_spam: np.ndarray       # [n] bool, the true labels
    wrong: np.ndarray         # [n] bool, misclassified at this step
    examples: tuple           # TokenRows for the token panel
    group_shares: np.ndarray  # [groups, experts] layer-1 share of each token group per expert


def clear_frames(frame_dir):
    """Delete frames left over from an earlier run, and nothing else."""
    os.makedirs(frame_dir, exist_ok=True)
    for ext in ("jpg", "jpeg", "png"):
        for old in glob.glob(os.path.join(frame_dir, f"*.{ext}")):
            os.remove(old)


def frame_path(frame_dir, step):
    return os.path.join(frame_dir, f"{step:07d}.jpg")


def _curve(ax, steps, values, color, label):
    """The history as a line, with a dot on the latest point (visible even when it is alone)."""
    ax.plot(steps, values, color=color, lw=1.5, label=label)
    ax.plot(steps[-1:], values[-1:], "o", color=color, ms=4)


class TrainingRecorder:
    """
    log() every evaluated step, keep() the steps to draw, then render() once at the end.
    With frame_dir=None it only keeps the history (for runs without a video).
    """

    def __init__(self, frame_dir, title, total_steps, n_experts, n_spam, n_ham, group_labels):
        self.frame_dir = frame_dir
        self.title = title
        self.total_steps = total_steps
        self.n_experts = n_experts
        self.n_spam = n_spam
        self.n_ham = n_ham
        self.group_labels = tuple(group_labels)
        self.steps = []
        self.losses = {}
        self.errors = {}
        self.loads = []   # one tuple per logged step: each layer's [experts] load
        self.extras = {}  # logged for the CSV, not drawn
        self.kept = []    # (step, number of logged steps at that point, FrameState)
        if frame_dir is not None:
            clear_frames(frame_dir)

    def log(self, step, losses, errors, loads, extras=None):
        self.steps.append(step)
        for k, v in losses.items():
            self.losses.setdefault(k, []).append(float(v))
        for k, v in errors.items():
            self.errors.setdefault(k, []).append(int(v))
        self.loads.append(tuple(np.array(layer, dtype=float) for layer in loads))
        for k, v in (extras or {}).items():
            self.extras.setdefault(k, []).append(float(v))

    def history_rows(self):
        """Everything logged so far, one dict per logged step, for saving as a CSV."""
        return [
            {"step": step,
             **{k: v[i] for k, v in self.losses.items()},
             **{k: v[i] for k, v in self.errors.items()},
             **{k: v[i] for k, v in self.extras.items()},
             **{f"layer {layer + 1} load expert {e + 1}": float(load[e])
                for layer, load in enumerate(self.loads[i]) for e in range(len(load))}}
            for i, step in enumerate(self.steps)
        ]

    def keep(self, step, state):
        """Remember what the model did at the step just logged; render() draws it."""
        if self.frame_dir is None:
            raise RuntimeError("this recorder keeps history only (it was made with frame_dir=None)")
        if not self.steps or self.steps[-1] != step:
            raise RuntimeError(f"log() step {step} before keeping its frame")
        self.kept.append((step, len(self.steps), state))

    def render(self):
        """Draw every kept step as frame_dir/<step>.jpg. Returns the number of frames."""
        half_width = ternary_view([state.cls_probs for _, _, state in self.kept])
        for step, n_logged, state in self.kept:
            with plt.rc_context(STYLE):
                fig = plt.figure(figsize=(FIGURE_PX[0] / DPI, FIGURE_PX[1] / DPI), dpi=DPI)
                fig.text(0.015, 0.955,
                         f"{self.title}   |   step {step:,} of {self.total_steps:,}",
                         fontsize=16, color=INK, weight="bold")
                self._draw_routing(fig, state, half_width)
                self._draw_curves(fig, n_logged)
                fig.savefig(frame_path(self.frame_dir, step), pil_kwargs={"quality": 92})
                plt.close(fig)
        return len(self.kept)

    def _draw_routing(self, fig, state, half_width):
        ax = fig.add_axes([0.01, 0.06, 0.43, 0.82])
        draw_ternary(ax, state.cls_probs, state.is_spam, state.wrong, half_width)
        ax.set_title("Layer 2 router: where each test message's [CLS] token is sent",
                     fontsize=12, loc="center")
        zoom = "" if half_width is None else "   (zoomed in on the centre)"
        fig.text(0.03, 0.025, f"{int(state.wrong.sum())} of {len(state.wrong):,} test messages "
                 f"misclassified{zoom}", fontsize=11, color=INK_2)

        ax = fig.add_axes([0.46, 0.55, 0.29, 0.33])
        draw_tokens(ax, state.examples)
        ax.set_title("Layer 1: the expert that processes each token", fontsize=12)

        fig.text(0.46, 0.455, "Layer 1: where each kind of token goes", fontsize=12, color=INK)
        fig.text(0.46, 0.43, "share of each group's test tokens sent to each expert",
                 fontsize=10, color=INK_2)
        ax = fig.add_axes([0.575, 0.04, 0.17, 0.32])
        draw_shares(ax, state.group_shares, self.group_labels)

    def _draw_curves(self, fig, n):
        """The history up to the n-th logged step, on axes that span the whole run."""
        steps = self.steps[:n]
        ax = fig.add_axes([0.79, 0.66, 0.195, 0.22])
        for (name, values), color in zip(self.losses.items(), (MUTED, INK), strict=False):
            _curve(ax, steps, values[:n], color, name)
        ax.set_yscale("log")
        ax.set_ylim(*LOSS_LIMITS)
        ax.set_title("Loss (binary cross-entropy)")
        self._finish_curve_axes(ax)

        ax = fig.add_axes([0.79, 0.36, 0.195, 0.22])
        for (name, values), color in zip(self.errors.items(), (INK, MUTED), strict=False):
            _curve(ax, steps, values[:n], color, name)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, 1.3 * max(self.n_spam, self.n_ham))
        ax.set_title(f"Test errors ({self.n_spam:,} spam, {self.n_ham:,} ham)")
        self._finish_curve_axes(ax)

        for layer, bottom in enumerate((0.175, 0.06)):
            ax = fig.add_axes([0.79, bottom, 0.195, 0.085])
            draw_load(ax, steps, [load[layer] for load in self.loads[:n]], self.total_steps)
            ax.set_ylabel(f"layer {layer + 1}", rotation=0, ha="right", va="center")
            if layer == 0:
                ax.set_title("Share of tokens per expert")
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("training step   (dashed: an even split)")

    def _finish_curve_axes(self, ax):
        ax.set_xlim(0, self.total_steps)
        ax.grid(color=GRID, lw=0.8)
        ax.legend(fontsize=9, loc="upper right", frameon=False, labelcolor=INK_2)
