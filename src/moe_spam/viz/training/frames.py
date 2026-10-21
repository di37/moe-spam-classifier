"""The training video's frames: one per kept step, in a layout that never moves."""

from moe_spam.viz.canvas import Canvas
from moe_spam.viz.panels import draw_curve, draw_load, draw_shares, draw_tokens
from moe_spam.viz.style import GRID, INK, INK_2, MUTED
from moe_spam.viz.ternary import Ternary
from moe_spam.viz.video import VIDEO_ENCODER, FrameWriter

TITLE = "Tiny BERT + mixture of experts, learning ham vs spam"
LOSS_LIMITS = (1e-3, 2.0)


def balancing_label(aux_weight):
    """Says whether a run used the load-balancing loss, and with what weight."""
    if aux_weight == 0:
        return "without load balancing (aux weight 0)"
    return f"with load balancing (aux weight {aux_weight:g})"


class TrainingFrames:
    """Draws a training run's kept steps, all sharing one zoom on the routing triangle.

    Attributes:
        history: The run's ``History``.
        total_steps: The steps of the whole run (every curve's x axis).
        n_spam: Spam in the test set.
        n_ham: Ham in the test set.
        group_labels: The heatmap's row labels.
        aux_weight: The weight of the run's load-balancing loss (0: none), named in the title.
    """

    def __init__(self, history, total_steps, n_spam, n_ham, group_labels, aux_weight):
        """Creates the frames of a run; see the attributes."""
        self.history = history
        self.total_steps = total_steps
        self.n_spam = n_spam
        self.n_ham = n_ham
        self.group_labels = tuple(group_labels)
        self.aux_weight = aux_weight

    @classmethod
    def of(cls, run):
        """Returns the frames of a ``TrainingRun``."""
        data = run.data
        return cls(run.history, run.config.steps, data.n_spam, data.n_ham,
                   data.groups.labels(data.test_ids), run.config.aux_weight)

    def render(self, frame_dir):
        """Draws every kept step as ``frame_dir/<step>.jpg``.

        Args:
            frame_dir: The folder of the frames; earlier frames in it are deleted first.

        Returns:
            The number of frames.
        """
        kept = self.history.kept
        half_width = Ternary.view([state.cls_probs for _, _, state in kept])
        writer = FrameWriter(frame_dir)
        writer.clear()
        title = f"{TITLE}   |   {balancing_label(self.aux_weight)}"
        for step, n_logged, state in kept:
            with Canvas(writer.path(step)) as canvas:
                canvas.title(title, f"step {step:,} of {self.total_steps:,}")
                self._draw_routing(canvas, state, half_width)
                self._draw_curves(canvas, n_logged)
        return len(kept)

    @staticmethod
    def encode(frame_dir, out_base):
        """Turns the frames into a video and a GIF.

        Args:
            frame_dir: The folder of the frames.
            out_base: The output path without extension: ``out_base.mp4`` and ``.gif``.
        """
        VIDEO_ENCODER.encode(frame_dir, out_base)

    def _draw_routing(self, canvas, state, half_width):
        ax = canvas.axes([0.01, 0.06, 0.43, 0.82])
        Ternary.draw(ax, state.cls_probs, state.is_spam, state.wrong, half_width)
        ax.set_title("Layer 2 router: where each test message's [CLS] token is sent",
                     fontsize=12, loc="center")
        zoom = "" if half_width is None else "   (zoomed in on the centre)"
        canvas.text(0.03, 0.025, f"{int(state.wrong.sum())} of {len(state.wrong):,} test "
                    f"messages misclassified{zoom}", fontsize=11, color=INK_2)

        ax = canvas.axes([0.46, 0.55, 0.29, 0.33])
        draw_tokens(ax, state.examples)
        ax.set_title("Layer 1: the expert that processes each token", fontsize=12)

        canvas.text(0.46, 0.455, "Layer 1: where each kind of token goes", fontsize=12,
                    color=INK)
        canvas.text(0.46, 0.43, "share of each group's test tokens sent to each expert",
                    fontsize=10, color=INK_2)
        ax = canvas.axes([0.575, 0.04, 0.17, 0.32])
        draw_shares(ax, state.group_shares, self.group_labels)

    def _draw_curves(self, canvas, n):
        """The history up to the n-th logged step, on axes that span the whole run."""
        history = self.history
        steps = history.steps[:n]
        ax = canvas.axes([0.79, 0.66, 0.195, 0.22])
        for (name, values), color in zip(history.losses.items(), (MUTED, INK), strict=False):
            draw_curve(ax, steps, values[:n], color, name)
        ax.set_yscale("log")
        ax.set_ylim(*LOSS_LIMITS)
        ax.set_title("Loss (binary cross-entropy)")
        self._finish_curve_axes(ax)

        ax = canvas.axes([0.79, 0.36, 0.195, 0.22])
        for (name, values), color in zip(history.errors.items(), (INK, MUTED), strict=False):
            draw_curve(ax, steps, values[:n], color, name)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, 1.3 * max(self.n_spam, self.n_ham))
        ax.set_title(f"Test errors ({self.n_spam:,} spam, {self.n_ham:,} ham)")
        self._finish_curve_axes(ax)

        for layer, bottom in enumerate((0.175, 0.06)):
            ax = canvas.axes([0.79, bottom, 0.195, 0.085])
            draw_load(ax, steps, [load[layer] for load in history.loads[:n]], self.total_steps)
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
