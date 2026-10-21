"""
Animate how the trained spam classifier reads a message, step by step, on one real spam and
one real ham message from the test set.

    python architecture_video.py                            # the model in checkpoints/moe.pt
    python architecture_video.py checkpoints/moe_noaux.pt   # any 2-layer, 3-expert checkpoint

The left of every frame is a map of the model, with layer 1 and layer 2 boxed; the right
shows what the lit-up step does to the message, using the model's own attention weights and
routing. Frames go to frames/architecture/, the video to media/architecture.{mp4,gif}.
"""
import argparse
import shutil
import textwrap

import matplotlib

matplotlib.use("Agg")  # render off-screen; frames are written to disk
import matplotlib.pyplot as plt

from architecture_panels import (
    draw_attention,
    draw_back,
    draw_classifier,
    draw_embed,
    draw_map,
    draw_route,
    draw_router_triangle,
    draw_text,
    draw_tokenize,
    panel_grid,
    token_layout,
)
from checkpoint import load_checkpoint
from dataset import prepare_data
from frames import DPI, FIGURE_PX, STYLE, clear_frames, frame_path
from panels import INK, INK_2
from video_creator import make_video
from vocab import LONG_NUMBER
from walkthrough import pick_messages

FPS = 15
FRAME_DIR = "frames/architecture"
VIDEO = "media/architecture"
GIF_WIDTH = 800
TITLE = "How the spam classifier reads a message"
CAPTION_CHARS = 112
LAYERS, EXPERTS = 2, 3  # the map shows two layers; the triangle and boxes three experts
CLOSE_CALL = 0.03  # say so when the router's top two probabilities for [CLS] are this close
DETAIL_RECT = [0.33, 0.03, 0.655, 0.75]  # where the step's drawing goes, in figure coordinates
# (step, seconds of animation, seconds held on its last frame). The ham message skips the
# embeddings and goes faster: the steps are the same.
BEATS = {
    "spam": (("text", 1.0, 1.5), ("tokenize", 2.0, 1.5), ("embed", 1.5, 1.5),
             ("attn1", 1.5, 2.0), ("route1", 2.0, 2.5), ("back1", 1.0, 0.5),
             ("attn2", 1.5, 1.5), ("route2", 2.0, 3.5), ("classify", 2.5, 3.0)),
    "ham": (("text", 0.7, 1.0), ("tokenize", 1.0, 0.8), ("attn1", 1.0, 1.0),
            ("route1", 1.5, 1.5), ("back1", 0.7, 0.3), ("attn2", 1.0, 0.8),
            ("route2", 1.5, 3.0), ("classify", 2.0, 1.0)),
}
STEP_NAMES = {
    "text": "A message comes in",
    "tokenize": "Tokenizer",
    "embed": "Embeddings",
    "attn1": "Layer 1: self-attention",
    "route1": "Layer 1: router and experts",
    "back1": "Layer 1: router and experts",
    "attn2": "Layer 2: self-attention",
    "route2": "Layer 2: router and experts",
    "classify": "Classifier",
}
CAPTIONS = {
    "text": "A real text message from the test set: the model never saw it during training.",
    "tokenize": "The text is cut into tokens: lowercase words, numbers and punctuation. Numbers "
                f"become [NUM] (up to {LONG_NUMBER - 1} digits) or [NUM5+]. A [CLS] token goes "
                "in front: it will collect a summary of the whole message.",
    "embed": "Each token becomes a list of {d_model} numbers (8 shown, darker = larger): a "
             "learned vector for the token plus one for its position.",
    "attn1": "Layer 1 starts with self-attention: every token takes in information from all the "
             "others. The bars show how much the [CLS] token takes from each one (averaged "
             "over the attention heads).",
    "route1": "Then the router sends each token to one of 3 experts, small feed-forward "
              "networks, and only that expert works on it. The training video's token panels "
              "show these choices.",
    "back1": "Each expert's output is added back to its token, and the updated tokens move on "
             "to layer 2.",
    "attn2": "Layer 2 repeats both steps with its own weights. Self-attention again: [CLS] "
             "gathers from the updated tokens.",
    "classify": "Only the final [CLS] is read: a LayerNorm, one linear layer and a sigmoid turn "
                "its {d_model} numbers into p(spam). Above 0.5 means spam.",
}
SCENES = {
    "text": draw_text,
    "tokenize": draw_tokenize,
    "embed": draw_embed,
    "attn1": lambda ax, trace, layout, t: draw_attention(ax, trace, layout, t, layer=0),
    "route1": lambda ax, trace, layout, t: draw_route(ax, trace, layout, t, layer=0),
    "back1": draw_back,
    "attn2": lambda ax, trace, layout, t: draw_attention(ax, trace, layout, t, layer=1),
    "route2": lambda ax, trace, layout, t: draw_route(ax, trace, layout, t, layer=1),
    "classify": draw_classifier,
}


def beats_for(label, fps=FPS):
    """[(step, animated frames, held frames)] for the spam or the ham message."""
    return [(step, max(1, round(seconds * fps)), round(hold * fps))
            for step, seconds, hold in BEATS[label]]


def step_titles(label):
    """Each step's title, numbered in the order this message shows them."""
    numbers = {}
    for step, *_ in BEATS[label]:
        numbers.setdefault(STEP_NAMES[step], len(numbers) + 1)
    return {step: f"{numbers[STEP_NAMES[step]]}. {STEP_NAMES[step]}" for step, *_ in BEATS[label]}


def caption(step, pick, label):
    d_model = pick.trace.embeddings.shape[1]
    if step != "route2":
        return CAPTIONS[step].format(d_model=d_model)
    expert = pick.trace.experts[-1][0] + 1
    how_many = (f"all {pick.total}" if pick.same == pick.total
                else f"{pick.same} of the {pick.total}")
    text = (f"Layer 2 repeats the router step. The training video's triangle plots one of its "
            f"choices: where each message's [CLS] goes. This one goes to expert {expert}, as "
            f"does the [CLS] of {how_many} {label} messages in the test set.")
    first, second = sorted(pick.trace.cls_probs[-1], reverse=True)[:2]
    if first - second < CLOSE_CALL:
        text += (f" A close call: the router gives {first:.3f} to expert {expert} and "
                 f"{second:.3f} to the next.")
    return text


def panel_size():
    """The detail panel's size in characters and lines (it only depends on the layout)."""
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(FIGURE_PX[0] / DPI, FIGURE_PX[1] / DPI), dpi=DPI)
        size = panel_grid(fig.add_axes(DETAIL_RECT))
        plt.close(fig)
    return size


def layout_for(trace, size):
    """Where every token goes on screen; ValueError if the message does not fit."""
    return token_layout(trace.text, trace.words, trace.experts, *size)


def fits_on_screen(trace, size=None):
    try:
        layout_for(trace, size or panel_size())
    except ValueError:
        return False
    return True


def draw_frame(path, pick, label, number, total, step, t, layout):
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(FIGURE_PX[0] / DPI, FIGURE_PX[1] / DPI), dpi=DPI)
        fig.text(0.015, 0.955, TITLE, fontsize=16, color=INK, weight="bold")
        fig.text(0.985, 0.955, f"message {number} of {total}: {label}", ha="right",
                 fontsize=13, color=INK_2)
        draw_map(fig.add_axes([0.01, 0.02, 0.27, 0.9]), step, pick.trace.embeddings.shape[1])
        fig.text(0.33, 0.885, step_titles(label)[step], fontsize=15, color=INK, weight="bold")
        fig.text(0.33, 0.862, "\n".join(textwrap.wrap(caption(step, pick, label), CAPTION_CHARS)),
                 fontsize=12, color=INK_2, va="top")
        ax = fig.add_axes(DETAIL_RECT)
        panel_grid(ax)
        SCENES[step](ax, pick.trace, layout, t)
        if step == "route2" and t >= 1:  # once every token has landed
            draw_router_triangle(fig.add_axes([0.80, 0.45, 0.17, 0.22]),
                                 pick.trace.cls_probs[-1])
        fig.savefig(path, pil_kwargs={"quality": 92})
        plt.close(fig)


def render_walkthrough(picks, frame_dir, fps=FPS):
    """picks: {"spam": Pick, "ham": Pick}. Draws every frame; returns how many."""
    if any(len(p.trace.experts) != LAYERS or len(p.trace.cls_probs[-1]) != EXPERTS
           for p in picks.values()):
        raise ValueError(f"the architecture video needs a model with {LAYERS} layers of "
                         f"{EXPERTS} experts")
    size = panel_size()
    layouts = {label: layout_for(pick.trace, size) for label, pick in picks.items()}
    clear_frames(frame_dir)  # only once everything above has been checked
    index = 0
    for number, (label, pick) in enumerate(picks.items(), start=1):
        for step, frames, hold in beats_for(label, fps):
            for i in range(frames):
                draw_frame(frame_path(frame_dir, index), pick, label, number, len(picks), step,
                           (i + 1) / frames, layouts[label])
                index += 1
            for _ in range(hold):  # pause: repeat the last frame
                shutil.copyfile(frame_path(frame_dir, index - 1), frame_path(frame_dir, index))
                index += 1
    return index


def main(args):
    model, vocab = load_checkpoint(args.checkpoint)
    if (model.config.n_layers, model.config.n_experts) != (LAYERS, EXPERTS):
        raise SystemExit(f"{args.checkpoint} has {model.config.n_layers} layers of "
                         f"{model.config.n_experts} experts; the architecture video needs "
                         f"{LAYERS} layers of {EXPERTS}")
    size = panel_size()
    picks = pick_messages(model, vocab, list(prepare_data().test_messages),
                          fits=lambda trace: fits_on_screen(trace, size))
    for label, pick in picks.items():
        print(f"{label}: {pick.text!r}  p(spam) = {pick.trace.p_spam:.3f}")
    n = render_walkthrough(picks, FRAME_DIR, args.fps)
    print(f"drew {n} frames")
    if not args.no_video:
        make_video(FRAME_DIR, VIDEO, fps=args.fps, gif_width=GIF_WIDTH, gif_every=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/moe.pt")
    parser.add_argument("--fps", type=int, default=FPS, choices=range(1, 61), metavar="FPS")
    parser.add_argument("--no-video", action="store_true", help="only draw the frames")
    main(parser.parse_args())
