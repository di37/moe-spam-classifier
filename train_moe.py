"""
Train a tiny BERT whose feed-forward layers are mixtures of experts to tell spam from ham,
and film it.

    python train_moe.py                          # 3 experts, with load balancing
    python train_moe.py --aux-weight 0           # no load-balancing loss
    python train_moe.py --experts 1 --no-video   # a plain dense BERT, for comparison

Every --frame-every steps (and at step 0, before any training) it runs the model on the same
held-out messages, so two frames differ only because the weights changed. After training it
draws a frame for each of those steps and turns them into media/training_<run>.{mp4,gif}.
Frames go to frames/training_<run>/, the model to checkpoints/<run>.pt and the run record to
results/training_<run>.{csv,json}. <run> is moe, moe_noaux or dense, followed by any setting
that differs from its default (e.g. moe_steps400_seed2).
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from bert import BertConfig, TinyBert
from checkpoint import save_checkpoint
from dataset import MAX_LEN, SPLIT_SEED, prepare_data
from frames import FrameState, TokenRow, TrainingRecorder
from run_log import write_training_record
from video_creator import make_video
from vocab import GROUPS, NUMBER_TOKENS, PAD, UNK, pieces, tokenize

# The frames, the checkpoint and every reported number use a moving average of the weights,
# which changes more smoothly from frame to frame than the raw weights do.
EMA_DECAY = 0.99
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
VIDEO_EXPERTS = 3       # the routing plot is a triangle, one corner per expert
EXAMPLE_TOKENS = {False: (8, 13), True: (8, 18)}  # token counts of the ham / spam examples
GIF_WIDTH = 800         # wide enough to read the token panel
PRINT_EVERY_FRAMES = 25
NAMED_SETTINGS = ("experts", "aux_weight", "steps", "batch_size", "lr", "seed")
TITLE = "Tiny BERT + mixture of experts, learning ham vs spam"


def pick_examples(messages, vocab):
    """
    Indices of the first ham and the first spam (one with a number in it) that are short
    enough to show whole and have no [UNK] token.
    """
    def fits(message):
        tokens = tokenize(message.text)
        low, high = EXAMPLE_TOKENS[message.is_spam]
        return (low <= len(tokens) <= high and all(vocab.id(t) != UNK for t in tokens)
                and (not message.is_spam or any(t in NUMBER_TOKENS for t in tokens)))

    picks = [next((i for i, m in enumerate(messages) if m.is_spam == is_spam and fits(m)), None)
             for is_spam in (False, True)]
    if None in picks:
        raise ValueError("no suitable ham and spam example message in the test set")
    return tuple(picks)


def group_shares(groups, experts, n_groups, n_experts):
    """[n_groups, n_experts]: of each group's tokens, the share each expert ran (NaN: none)."""
    counts = torch.bincount(groups * n_experts + experts, minlength=n_groups * n_experts)
    counts = counts.reshape(n_groups, n_experts).double()
    return (counts / counts.sum(dim=1, keepdim=True)).numpy()


def group_labels(data):
    """Row labels for the heatmap: each group's name and how many test tokens it has."""
    real = data.test_ids != PAD
    counts = torch.bincount(data.token_group[data.test_ids][real], minlength=len(GROUPS))
    return [f"{name} ({count:,})" for name, count in zip(GROUPS, counts.tolist(), strict=True)]


def _token_row(message, experts, p_spam):
    words = ("[CLS]", *pieces(message.text))[:MAX_LEN]
    return TokenRow("spam" if message.is_spam else "ham", words,
                    tuple(experts[:len(words)].tolist()), float(p_spam))


@torch.no_grad()
def evaluate(model, data, examples):
    """
    Run the model in eval mode on the whole test set, and on as many training messages for a
    training loss measured the same way: (metrics, FrameState, each layer's expert load).
    """
    was_training = model.training
    model.eval()
    out = model(data.test_ids)
    n_train = len(data.test_ids)  # the training half is in random order: a random sample
    train_logits = model(data.train_ids[:n_train]).logits
    model.train(was_training)

    labels = data.test_labels.bool()
    wrong = (out.logits > 0) != labels
    first = out.routing[0]
    p_spam = torch.sigmoid(out.logits)
    state = FrameState(
        cls_probs=out.routing[-1].probs[:, 0].numpy().copy(),  # a copy, not a view of [n, L, E]
        is_spam=labels.numpy(),
        wrong=wrong.numpy(),
        examples=tuple(_token_row(data.test_messages[i], first.expert[i], p_spam[i])
                       for i in examples),
        group_shares=group_shares(data.token_group[data.test_ids][first.mask],
                                  first.expert[first.mask], len(GROUPS), model.config.n_experts),
    )
    metrics = {
        "train BCE": F.binary_cross_entropy_with_logits(
            train_logits, data.train_labels[:n_train]).item(),
        "test BCE": F.binary_cross_entropy_with_logits(out.logits, data.test_labels).item(),
        "errors": int(wrong.sum()),
        "spam missed": int((wrong & labels).sum()),
        "ham flagged": int((wrong & ~labels).sum()),
    }
    return metrics, state, [r.load().numpy() for r in out.routing]


def summarise(metrics, state, loads):
    """The final numbers for the run record and the README."""
    n_spam = int(state.is_spam.sum())
    caught = n_spam - metrics["spam missed"]
    top = state.cls_probs.argmax(axis=1)
    experts = range(state.cls_probs.shape[1])
    by_class = {label: [round(float(np.mean(top[state.is_spam == is_spam] == e)), 4)
                        for e in experts]
                for label, is_spam in (("ham", False), ("spam", True))}
    last = len(loads)
    return {
        **metrics,
        "accuracy": 1 - metrics["errors"] / len(state.is_spam),
        "spam F1": 2 * caught / (2 * caught + metrics["ham flagged"] + metrics["spam missed"]),
        **{f"layer {i + 1} load": [round(float(v), 4) for v in load]
           for i, load in enumerate(loads)},
        f"layer {last} [CLS] top-1 expert, ham messages": by_class["ham"],
        f"layer {last} [CLS] top-1 expert, spam messages": by_class["spam"],
    }


def record_frame(recorder, model, data, examples, step, batch_bce, aux_loss):
    """Evaluate, log the curves, keep the frame (when filming) and print now and then."""
    metrics, state, loads = evaluate(model, data, examples)
    recorder.log(step, {"train BCE": metrics["train BCE"], "test BCE": metrics["test BCE"]},
                 {"spam missed": metrics["spam missed"], "ham flagged": metrics["ham flagged"]},
                 loads, {"batch BCE": batch_bce, "aux loss": aux_loss})
    if recorder.frame_dir is not None:
        recorder.keep(step, state)
    if len(recorder.steps) % PRINT_EVERY_FRAMES == 1 or step == recorder.total_steps:
        print(f"step {step:5d}  train BCE {metrics['train BCE']:.4f}  test BCE "
              f"{metrics['test BCE']:.4f}  errors {metrics['errors']:3d}  load "
              + " | ".join(" ".join(f"{v:.2f}" for v in load) for load in loads), flush=True)


def lr_factor(total_steps, warmup=WARMUP_STEPS):
    """Linear warmup to the full rate, then linear decay towards 0 at the last step."""
    warmup = min(warmup, total_steps)

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    return factor


def build(args, vocab_size):
    """The model, its moving average, the optimiser and the learning-rate schedule."""
    torch.manual_seed(args.seed)
    model = TinyBert(BertConfig(vocab_size=vocab_size, max_len=MAX_LEN,
                                n_experts=args.experts)).train()
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    return model, ema, opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_factor(args.steps))


def train_loop(args, data, examples, recorder):
    """Step 0 records the untrained model; each later step is one update. Returns the EMA model."""
    model, ema, opt, schedule = build(args, len(data.vocab))
    batches = torch.Generator().manual_seed(args.seed)
    record_frame(recorder, ema.module, data, examples, 0, math.nan, math.nan)
    bce_sum, aux_sum, count = 0.0, 0.0, 0
    for step in range(1, args.steps + 1):
        rows = torch.randint(0, len(data.train_ids), (args.batch_size,), generator=batches)
        out = model(data.train_ids[rows])
        bce = F.binary_cross_entropy_with_logits(out.logits, data.train_labels[rows])
        opt.zero_grad()
        (bce + args.aux_weight * out.aux_loss).backward()
        opt.step()
        schedule.step()
        ema.update_parameters(model)
        bce_sum, aux_sum, count = bce_sum + bce.item(), aux_sum + out.aux_loss.item(), count + 1
        if step % args.frame_every == 0 or step == args.steps:
            record_frame(recorder, ema.module, data, examples, step, bce_sum / count,
                         aux_sum / count)
            bce_sum, aux_sum, count = 0.0, 0.0, 0
    return ema.module


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def train(args):
    start = time.perf_counter()
    name = run_name(args)
    data = prepare_data()
    examples = pick_examples(data.test_messages, data.vocab)
    n_spam = int(data.test_labels.sum())
    frame_dir = None if args.no_video else f"frames/training_{name}"
    recorder = TrainingRecorder(frame_dir, TITLE, args.steps, args.experts, n_spam,
                                len(data.test_labels) - n_spam, group_labels(data))
    timings = {"data_preparation": time.perf_counter() - start}
    phase_start = time.perf_counter()
    model = train_loop(args, data, examples, recorder)
    timings["training_and_evaluation"] = time.perf_counter() - phase_start

    final = summarise(*evaluate(model, data, examples))
    checkpoint = f"checkpoints/{name}.pt"
    save_checkpoint(checkpoint, model, data.vocab)
    seeds = {"training": args.seed, "split": SPLIT_SEED}

    def write_record():
        write_training_record(f"training_{name}", args, seeds, recorder, final, timings,
                              checkpoint, frames=len(recorder.kept))

    write_record()  # before the video, so a failed encode does not lose the results
    if frame_dir is not None:
        timings["drawing_frames"] = _timed(recorder.render)
        timings["video_encoding"] = _timed(lambda: make_video(
            frame_dir, f"media/training_{name}", gif_width=GIF_WIDTH, gif_every=2))
    timings["total"] = time.perf_counter() - start
    write_record()
    return final


def run_name(args):
    """moe, moe_noaux or dense, then every training setting that differs from its default."""
    base = "dense" if args.experts == 1 else ("moe_noaux" if args.aux_weight == 0 else "moe")
    defaults = build_parser().parse_args([])
    in_base = {"experts": (1, defaults.experts), "aux_weight": (0, defaults.aux_weight)}
    changed = [f"{key.replace('_', '')}{getattr(args, key)}" for key in NAMED_SETTINGS
               if getattr(args, key) != getattr(defaults, key)
               and getattr(args, key) not in in_base.get(key, ())]
    return "_".join([base, *changed])


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _float_at_least(low, inclusive):
    def parse(text):
        value = float(text)
        if not math.isfinite(value) or value < low or (value == low and not inclusive):
            raise argparse.ArgumentTypeError(
                f"must be a number {'at least' if inclusive else 'more than'} {low}, got {text}")
        return value
    return parse


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=positive_int, default=800, help="training steps")
    parser.add_argument("--batch-size", type=positive_int, default=64, help="messages per step")
    parser.add_argument("--lr", type=_float_at_least(0, inclusive=False), default=1e-3,
                        help="peak AdamW learning rate")
    parser.add_argument("--experts", type=positive_int, default=3,
                        help="experts per MoE layer (1 = a plain dense BERT)")
    parser.add_argument("--aux-weight", type=_float_at_least(0, inclusive=True), default=0.01,
                        help="weight of the load-balancing loss (0 = off)")
    parser.add_argument("--frame-every", type=positive_int, default=5,
                        help="evaluate and draw a frame every N steps (and at the last step)")
    parser.add_argument("--seed", type=int, default=0, help="seed for weights and batches")
    parser.add_argument("--no-video", action="store_true",
                        help="only train and write the run record; no frames or video")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.experts != VIDEO_EXPERTS and not args.no_video:
        parser.error(f"the video's routing plot needs --experts {VIDEO_EXPERTS}; "
                     f"add --no-video to train with {args.experts}")
    return args


if __name__ == "__main__":
    train(parse_args())
