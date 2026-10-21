"""
Animate how a mixture of experts speeds up inference: it does a fraction of the work of a
dense layer with the same weights. Then show what that is worth in measured time.

    python inference_video.py           # measures the layers, then draws the video
    python inference_video.py --reuse   # uses the timings saved in results/ instead

Three versions of one feed-forward layer see the same tokens of a real test message: a dense
network, one 3 times wider (as many weights as the MoE layer), and the MoE layer, where the
trained model's layer-1 router sends each token to one expert. Counters add up the work each
layer does. Then three sets of timings measured on this computer (inference_benchmark.py):
many tokens at once (throughput), one message at a time (latency), and the whole trained
dense and MoE classifiers. Frames go to frames/inference/, the video to
media/inference.{mp4,gif}. Needs checkpoints/moe.pt and checkpoints/dense.pt.
"""
import argparse
import os
import shutil
import textwrap

import matplotlib

matplotlib.use("Agg")  # render off-screen; frames are written to disk
import matplotlib.pyplot as plt

from architecture_video import fits_on_screen, panel_size
from checkpoint import load_checkpoint
from dataset import prepare_data
from frames import DPI, FIGURE_PX, STYLE, clear_frames, frame_path
from inference_benchmark import (
    LAYERS,
    RECORD,
    SIZES,
    benchmark,
    layer_costs,
    record,
    time_models,
)
from inference_panels import draw_bars, draw_lanes, lane_labels
from panels import INK, INK_2
from run_log import read_json
from video_creator import make_video
from vocab import PAD, encode
from walkthrough import pick_messages

FPS = 15
FRAME_DIR = "frames/inference"
VIDEO = "media/inference"
GIF_WIDTH = 800
TITLE = "How a mixture of experts speeds up inference"
CAPTION_CHARS = 170
EXPERTS = 3
DENSE_CHECKPOINT = "checkpoints/dense.pt"
SAME_SPEED = 0.05  # timings within 5% of each other count as "as fast"
SLOW_TOKENS, SLOW_SHARE = 4, 0.5  # the first tokens take half the stream, then it speeds up
LIT_SHARE = 0.8  # each token's layers are lit for this much of its turn
# (scene, seconds of animation, seconds held on its last frame)
BEATS = (("setup", 1.0, 1.5), ("stream", 9.0, 0.5), ("totals", 1.0, 3.5),
         ("throughput", 1.0, 5.0), ("latency", 1.0, 5.0), ("models", 1.0, 5.0))
SCENE_TITLES = {"setup": "1. Three layers, the same tokens",
                "stream": "2. A message goes through",
                "totals": "3. Work done, weights held",
                "throughput": "4. Measured: many tokens at once",
                "latency": "5. Measured: one message at a time",
                "models": "6. Measured: the whole classifier"}
MODEL_NAMES = {"dense": "dense model (1 expert)", "moe": "MoE model (3 experts)"}
LANES_RECT = [0.015, 0.03, 0.6, 0.76]
TOP_CHART, BOTTOM_CHART = [0.70, 0.47, 0.27, 0.25], [0.70, 0.09, 0.27, 0.25]


def beats(fps=FPS):
    """[(scene, animated frames, held frames)]."""
    return [(scene, max(1, round(seconds * fps)), round(hold * fps))
            for scene, seconds, hold in BEATS]


def token_progress(t, n):
    """How many of n tokens have gone through at progress t in [0, 1], as a float."""
    slow = min(SLOW_TOKENS, n)
    if t <= SLOW_SHARE:
        return slow * t / SLOW_SHARE
    return min(n, slow + (n - slow) * (t - SLOW_SHARE) / (1 - SLOW_SHARE))


def tokens_done(t, n):
    return int(token_progress(t, n))


def compare(value, reference):
    """How a time compares with a reference time: "2.0× slower", "4.0× faster", "as fast"."""
    ratio = value / reference
    if abs(ratio - 1) < SAME_SPEED:
        return "as fast"
    return f"{ratio:.1f}× slower" if ratio > 1 else f"{1 / ratio:.1f}× faster"


def _captions(trace, costs, measured, n_experts):
    dense, wide, moe = (costs[k].macs_per_token for k in LAYERS)
    d_model = trace.embeddings.shape[1]
    n = len(trace.words)
    text = {
        "setup": f"Three versions of the same feed-forward layer, fed the same tokens (each a "
                 f"list of {d_model} numbers). The rest of the model would be identical.",
        "stream": f"For each token the dense layer does {dense:,} multiply-adds, and the one "
                  f"{n_experts}× wider {wide:,}. The MoE layer holds the same weights as the "
                  f"wide one but runs only the expert its router picks (here, the trained "
                  f"model's layer-1 choices), so it does {moe:,}.",
        "totals": f"After {n} tokens the MoE layer has done {moe / wide:.0%} of the wide layer's "
                  f"work, about as much as the small dense layer, yet it holds as many weights "
                  f"as the wide one. MoE saves compute, not memory.",
    }
    if not measured:
        return text
    sizes = measured["sizes"]
    ours, large = sizes["ours"]["ms_per_1k_tokens"], sizes["large"]["ms_per_1k_tokens"]
    text["throughput"] = (
        f"Measured on this computer, with many tokens at once ({sizes['ours']['batch_tokens']:,} "
        f"per batch), as a server busy with many messages would run. At this project's tiny "
        f"size the routing overhead eats the saving: the MoE layer is "
        f"{compare(ours['moe'], ours['wide'])} than the wide one (this varies a lot from run "
        f"to run). With large experts it is "
        f"{compare(large['moe'], large['wide'])} than a dense layer with the same weights "
        f"({sizes['large']['n_experts']}× would be perfect), and "
        f"{compare(large['moe'], large['dense'])} than one expert alone.")
    ours, large = sizes["ours"]["latency_us"], sizes["large"]["latency_us"]
    used = len(set(trace.experts[0]))
    used = f"all {n_experts}" if used == n_experts else f"{used} of the {n_experts}"
    text["latency"] = (
        f"One message of {sizes['ours']['latency_tokens']} tokens at a time, the way a spam filter "
        f"is called. Its tokens spread over almost every expert (this message uses {used} in "
        f"layer 1), so almost every expert's "
        f"weights must be read anyway, and each expert adds its own overhead. The MoE layer is "
        f"{compare(ours['moe'], ours['wide'])} than the wide one at this project's size, and "
        f"{compare(large['moe'], large['wide'])} at the large size. MoE's speed-up needs big "
        f"batches.")
    dense_model, moe_model = measured["models"]["dense"], measured["models"]["moe"]
    text["models"] = (
        f"The two trained classifiers, end to end: attention, embeddings and the classifier cost "
        f"the same in both. The MoE model does the dense model's work per token plus routing, so "
        f"here it can only lose: it is "
        f"{compare(moe_model['one_message_ms'], dense_model['one_message_ms'])} for one message, "
        f"and {compare(moe_model['ms_per_1k_messages'], dense_model['ms_per_1k_messages'])} for "
        f"all {measured['test_messages']:,} test messages in one batch.")
    return text


def _draw_layer_timings(fig, measured, scene):
    """Throughput or latency of the three layers, at this project's size and a large one."""
    for rect, key, name in ((TOP_CHART, "ours", "this project's size"),
                            (BOTTOM_CHART, "large", "a large layer")):
        size = measured["sizes"][key]
        names = {k: v[2] for k, v in lane_labels(size["d_ff"], size["n_experts"]).items()}
        shape = f"{size['d_model']:,} → {size['d_ff']:,}, {size['n_experts']} experts"
        if scene == "throughput":
            values, fmt = size["ms_per_1k_tokens"], (lambda value: f"{value:.2f} ms")
            title = f"Many tokens at once: {name}\n{shape} · ms per 1,000 tokens"
        else:
            values, fmt = size["latency_us"], (lambda value: f"{value:,.0f} µs")
            title = (f"One message ({size['latency_tokens']} tokens): {name}\n"
                     f"{shape} · microseconds")
        draw_bars(fig.add_axes(rect), values, names, 1.35 * max(values.values()), title, fmt)


def _draw_model_timings(fig, measured):
    models, keys = measured["models"], tuple(MODEL_NAMES)
    one = {k: models[k]["one_message_ms"] for k in keys}
    draw_bars(fig.add_axes(TOP_CHART), one, MODEL_NAMES, 1.35 * max(one.values()),
              "Whole classifier, one message\nmilliseconds", lambda v: f"{v:.2f} ms", keys)
    batch = {k: models[k]["ms_per_1k_messages"] for k in keys}
    draw_bars(fig.add_axes(BOTTOM_CHART), batch, MODEL_NAMES, 1.35 * max(batch.values()),
              f"Whole classifier, all {measured['test_messages']:,} test messages at once\n"
              f"milliseconds per 1,000 messages", lambda v: f"{v:.1f} ms", keys)


def draw_frame(path, scene, t, trace, costs, measured, d_ff, n_experts):
    n = len(trace.words)
    streaming = scene == "stream"
    done = {"setup": 0.0, "stream": token_progress(t, n)}.get(scene, float(n))
    current = min(int(done), n - 1)
    # A token counts as soon as it goes in, so the bars and the title always agree.
    counted = min(n, int(done) + 1) if streaming else int(done)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(FIGURE_PX[0] / DPI, FIGURE_PX[1] / DPI), dpi=DPI)
        fig.text(0.015, 0.955, TITLE, fontsize=16, color=INK, weight="bold")
        fig.text(0.985, 0.955, SCENE_TITLES[scene], ha="right", fontsize=13, color=INK_2)
        caption = _captions(trace, costs, measured, n_experts)[scene]
        fig.text(0.015, 0.905, "\n".join(textwrap.wrap(caption, CAPTION_CHARS)), fontsize=12,
                 color=INK_2, va="top")
        draw_lanes(fig.add_axes(LANES_RECT), trace.words[current] if streaming else None,
                   trace.experts[0][current], streaming and done - int(done) < LIT_SHARE,
                   costs, d_ff, n_experts)
        if scene in ("throughput", "latency"):
            _draw_layer_timings(fig, measured, scene)
        elif scene == "models":
            _draw_model_timings(fig, measured)
        else:
            names = {k: v[2] for k, v in lane_labels(d_ff, n_experts).items()}
            work = {k: counted * costs[k].macs_per_token for k in LAYERS}
            draw_bars(fig.add_axes(TOP_CHART), work, names,
                      1.35 * n * costs["wide"].macs_per_token,
                      f"Multiply-adds so far ({counted} of {n} tokens)")
            weights = {k: costs[k].weights for k in LAYERS}
            draw_bars(fig.add_axes(BOTTOM_CHART), weights, names, 1.35 * max(weights.values()),
                      "Weights held in memory")
        fig.savefig(path, pil_kwargs={"quality": 92})
        plt.close(fig)


def render_inference(trace, measured, frame_dir, fps=FPS, d_ff=None):
    """
    trace: a walkthrough.MessageTrace of the trained model; measured: what
    inference_benchmark.record saved. Draws every frame; returns how many.
    """
    if len(trace.cls_probs[0]) != EXPERTS:
        raise ValueError(f"the inference video needs a model with {EXPERTS} experts")
    d_model = trace.embeddings.shape[1]
    d_ff = d_ff or measured["sizes"]["ours"]["d_ff"]
    costs = {c.name: c for c in layer_costs(d_model, d_ff, EXPERTS)}
    clear_frames(frame_dir)
    index = 0
    for scene, frames, hold in beats(fps):
        for i in range(frames):
            draw_frame(frame_path(frame_dir, index), scene, (i + 1) / frames, trace, costs,
                       measured, d_ff, EXPERTS)
            index += 1
        for _ in range(hold):  # pause: repeat the last frame
            shutil.copyfile(frame_path(frame_dir, index - 1), frame_path(frame_dir, index))
            index += 1
    return index


def main(args):
    model, vocab = load_checkpoint(args.checkpoint)
    config = model.config
    if (config.d_model, config.d_ff, config.n_experts) != SIZES["ours"]:
        raise SystemExit(f"{args.checkpoint} is not this project's layer size {SIZES['ours']}")
    if not os.path.isfile(DENSE_CHECKPOINT):
        raise SystemExit(f"{DENSE_CHECKPOINT} not found: train the dense model first with "
                         f"python train_moe.py --experts 1 --no-video")
    dense, dense_vocab = load_checkpoint(DENSE_CHECKPOINT)
    if dense_vocab != vocab:
        raise SystemExit("the dense and MoE checkpoints were trained on different vocabularies")
    data = prepare_data()
    size = panel_size()
    pick = pick_messages(model, vocab, list(data.test_messages),
                         fits=lambda trace: fits_on_screen(trace, size))["spam"]
    measured = read_json(RECORD) if args.reuse and os.path.isfile(RECORD) else None
    if not measured or not measured.get("models"):
        print("measuring the layers and both classifiers (about a minute)...", flush=True)
        one = encode(vocab, [pick.text], config.max_len)
        one = one[:, :int((one != PAD).sum())]  # one message alone needs no padding
        batch = encode(vocab, [m.text for m in data.test_messages], config.max_len)
        measured = record(benchmark(), time_models({"dense": dense, "moe": model}, one, batch),
                          len(batch))
    print(f"message: {pick.text!r}")
    n = render_inference(pick.trace, measured, FRAME_DIR, args.fps, d_ff=config.d_ff)
    print(f"drew {n} frames")
    if not args.no_video:
        make_video(FRAME_DIR, VIDEO, fps=args.fps, gif_width=GIF_WIDTH, gif_every=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/moe.pt")
    parser.add_argument("--fps", type=int, default=FPS, choices=range(1, 61), metavar="FPS")
    parser.add_argument("--reuse", action="store_true",
                        help="use the timings in results/inference_benchmark.json")
    parser.add_argument("--no-video", action="store_true", help="only draw the frames")
    main(parser.parse_args())
