"""The commands that train and measure: ``train``, ``baselines`` and ``benchmark``."""

import time
from dataclasses import fields

from moe_spam.analysis.benchmark import Benchmark, saved_timings
from moe_spam.cli.common import Command, float_at_least, positive_int
from moe_spam.data.dataset import SpamData
from moe_spam.paths import PATHS
from moe_spam.records import RecordStore
from moe_spam.training.baselines import RECORD, RECORD_KEY, BagOfWordsBaseline, results_table
from moe_spam.training.config import TrainingConfig
from moe_spam.training.trainer import RECORD_PREFIX, Trainer
from moe_spam.viz.training.frames import TrainingFrames


class TrainCommand(Command):
    """``moe-spam train``: train a model and make its training video."""

    name = "train"
    help = "train the classifier and make its training video"

    def add_arguments(self, parser):
        """Adds one option per ``TrainingConfig`` setting, with its default."""
        defaults = TrainingConfig()
        parser.add_argument("--steps", type=positive_int, default=defaults.steps,
                            help="training steps")
        parser.add_argument("--batch-size", type=positive_int, default=defaults.batch_size,
                            help="messages per step")
        parser.add_argument("--lr", type=float_at_least(0, inclusive=False), default=defaults.lr,
                            help="peak AdamW learning rate")
        parser.add_argument("--experts", type=positive_int, default=defaults.experts,
                            help="experts per MoE layer (1 = a plain dense BERT)")
        parser.add_argument("--aux-weight", type=float_at_least(0, inclusive=True),
                            default=defaults.aux_weight,
                            help="weight of the load-balancing loss (0 = off)")
        parser.add_argument("--frame-every", type=positive_int, default=defaults.frame_every,
                            help="evaluate and draw a frame every N steps (and at the last step)")
        parser.add_argument("--seed", type=int, default=defaults.seed,
                            help="seed for weights and batches")
        parser.add_argument("--no-video", action="store_true",
                            help="only train and write the run record; no frames or video")

    @staticmethod
    def config(args, parser):
        """Returns the ``TrainingConfig`` of the parsed ``args`` (usage error if invalid)."""
        try:
            return TrainingConfig(**{f.name: getattr(args, f.name)
                                     for f in fields(TrainingConfig)})
        except ValueError as err:
            parser.error(str(err))

    def run(self, args, parser):
        """Trains; then, unless ``--no-video``, draws the frames and encodes the video.

        Returns:
            The trained model's final metrics.
        """
        paths, store = PATHS, RecordStore()
        run = Trainer(self.config(args, parser), paths=paths, store=store).run()
        if run.config.no_video:
            return run.final_metrics
        frame_dir = paths.frame_dir(run.name)
        frames = TrainingFrames.of(run)
        drawing = _timed(lambda: frames.render(frame_dir))
        encoding = _timed(lambda: frames.encode(frame_dir, paths.video(run.name)))
        total = run.timings["total"] + drawing + encoding
        run.with_timings(drawing_frames=drawing, video_encoding=encoding,
                         total=total).write_record(store)
        return run.final_metrics


class BaselinesCommand(Command):
    """``moe-spam baselines``: the bag-of-words baseline, and every run's results in a table."""

    name = "baselines"
    help = "fit the bag-of-words baseline and print every run's results"

    def add_arguments(self, parser):
        """This command has no options."""

    def run(self, args, parser):
        """Fits the baseline, records it, and prints it next to every training record."""
        baseline = BagOfWordsBaseline()
        scores = baseline.run(SpamData.prepare()).as_dict()
        store = RecordStore()
        store.write_json(RECORD, {RECORD_KEY: scores, **baseline.as_record()})
        runs = {"bag-of-words logreg": scores}
        for name, record in store.read_all(f"{RECORD_PREFIX}*").items():
            runs[name.removeprefix(RECORD_PREFIX)] = record["final_metrics"]
        print(results_table(runs))


class BenchmarkCommand(Command):
    """``moe-spam benchmark``: time the dense, wide and MoE layers."""

    name = "benchmark"
    help = "time the dense, wide and MoE feed-forward layers"

    def add_arguments(self, parser):
        """This command has no options."""

    def run(self, args, parser):
        """Times the layers at every size, records the results and prints them.

        Whole-model timings saved by ``inference-video`` are kept.
        """
        benchmark, store = Benchmark(), RecordStore()
        saved = saved_timings(store) or {}
        results = benchmark.layers()
        benchmark.record(results, saved.get("models"), saved.get("test_messages"), store)
        for name, r in results.items():
            many = "  ".join(f"{k} {v:.3f} ms" for k, v in r["ms_per_1k_tokens"].items())
            one = "  ".join(f"{k} {v:,.0f} us" for k, v in r["latency_us"].items())
            print(f"{name}: d_model {r['d_model']}, d_ff {r['d_ff']}, {r['n_experts']} experts\n"
                  f"  many tokens: {many} per 1,000 tokens\n"
                  f"  one message ({r['latency_tokens']} tokens): {one}")


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start
