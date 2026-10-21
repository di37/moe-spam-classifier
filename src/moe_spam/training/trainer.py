"""Training the tiny BERT with mixture-of-experts layers to tell spam from ham."""

import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from moe_spam.analysis.examples import ExampleFinder
from moe_spam.data.dataset import MAX_LEN, SPLIT_SEED, SpamData
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.model.checkpoint import Checkpoint
from moe_spam.paths import PATHS
from moe_spam.records import RecordStore
from moe_spam.training.config import EMA_DECAY, WARMUP_STEPS, WEIGHT_DECAY, TrainingConfig
from moe_spam.training.evaluator import Evaluator
from moe_spam.training.history import History

RECORD_PREFIX = "training_"  # results/training_<run name>.{csv,json}
PRINT_EVERY_FRAMES = 25


class WarmupLinearDecay:
    """A learning-rate factor: linear warmup to the full rate, then linear decay towards 0.

    Attributes:
        total_steps: The steps of the whole run.
        warmup: The warmup steps (at most ``total_steps``).
    """

    def __init__(self, total_steps, warmup=WARMUP_STEPS):
        """Creates the schedule for a run of ``total_steps`` steps."""
        self.total_steps = total_steps
        self.warmup = min(warmup, total_steps)

    def __call__(self, step):
        """Returns the factor of the learning rate at ``step`` (0-based)."""
        if step < self.warmup:
            return (step + 1) / self.warmup
        return max(0.0, (self.total_steps - step) / max(self.total_steps - self.warmup, 1))


@dataclass(frozen=True)
class TrainingRun:
    """A finished training run, before or after its video is made.

    Attributes:
        config: The run's settings.
        data: The data it was trained and evaluated on.
        history: What it logged.
        final_metrics: The trained model's numbers.
        timings: Seconds per phase.
        checkpoint: Where the trained model was saved.
    """

    config: TrainingConfig
    data: SpamData
    history: History
    final_metrics: dict
    timings: dict
    checkpoint: Path

    @property
    def name(self):
        """The record name: ``training_<run name>``."""
        return f"{RECORD_PREFIX}{self.config.run_name}"

    def with_timings(self, **seconds):
        """Returns a copy of this run with ``seconds`` added to its timings."""
        return dataclasses.replace(self, timings={**self.timings, **seconds})

    def write_record(self, store=None):
        """Writes ``<name>.csv`` (the logged history) and ``<name>.json`` (everything else)."""
        store = store or RecordStore()
        store.write_csv(self.name, self.history.rows())
        store.write_json(self.name, {
            "run": self.name,
            "command": store.command(),
            "args": self.config.as_record(),
            "seeds": {"training": self.config.seed, "split": SPLIT_SEED},
            "final_metrics": self.final_metrics,
            "logged_steps": len(self.history.steps),
            "frames": len(self.history.kept),
            "wall_clock_s": self.timings,
            "checkpoint": {"path": str(self.checkpoint),
                           "sha256": store.sha256(self.checkpoint)},
            "environment": store.environment(),
        })


class Trainer:
    """Trains one model, keeps its history, and saves the model and its record.

    Step 0 records the untrained model; every later step is one update. The frames, the
    checkpoint and every reported number use a moving average of the weights, which changes
    more smoothly from frame to frame than the raw weights.

    Attributes:
        config: The run's settings.
    """

    def __init__(self, config, data=None, paths=PATHS, store=None):
        """Creates a trainer.

        Args:
            config: A ``TrainingConfig``.
            data: The prepared data; by default the SMS Spam Collection, prepared on ``run``.
            paths: Where the checkpoint goes.
            store: Where the record goes (``results/`` by default).
        """
        self.config = config
        self._data = data
        self._paths = paths
        self._store = store or RecordStore()

    def run(self):
        """Trains, evaluates and saves; returns the ``TrainingRun``, its record written."""
        start = time.perf_counter()
        data = self._data or SpamData.prepare()
        evaluator = Evaluator(data, ExampleFinder(data.vocab).first(data.test_messages))
        history = History()
        timings = {"data_preparation": time.perf_counter() - start}
        phase = time.perf_counter()
        model = self._train(data, evaluator, history)
        timings["training_and_evaluation"] = time.perf_counter() - phase
        final = evaluator.evaluate(model).summary()
        checkpoint = self._paths.checkpoint(self.config.run_name)
        Checkpoint(model, data.vocab).save(checkpoint)
        run = TrainingRun(self.config, data, history, final,
                          {**timings, "total": time.perf_counter() - start}, checkpoint)
        run.write_record(self._store)  # now, so a failed video does not lose the results
        return run

    def _build(self, vocab_size):
        """Returns the model, its moving average, the optimiser and the schedule."""
        torch.manual_seed(self.config.seed)
        model = TinyBert(BertConfig(vocab_size=vocab_size, max_len=MAX_LEN,
                                    n_experts=self.config.experts)).train()
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY))
        opt = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=WEIGHT_DECAY)
        schedule = torch.optim.lr_scheduler.LambdaLR(opt, WarmupLinearDecay(self.config.steps))
        return model, ema, opt, schedule

    def _train(self, data, evaluator, history):
        """Runs the training loop; returns the moving-average model."""
        config = self.config
        model, ema, opt, schedule = self._build(len(data.vocab))
        batches = torch.Generator().manual_seed(config.seed)
        self._record(history, evaluator, ema.module, 0, math.nan, math.nan)
        bce_sum, aux_sum, count = 0.0, 0.0, 0
        for step in range(1, config.steps + 1):
            rows = torch.randint(0, len(data.train_ids), (config.batch_size,), generator=batches)
            out = model(data.train_ids[rows])
            bce = F.binary_cross_entropy_with_logits(out.logits, data.train_labels[rows])
            opt.zero_grad()
            (bce + config.aux_weight * out.aux_loss).backward()
            opt.step()
            schedule.step()
            ema.update_parameters(model)
            bce_sum, aux_sum, count = bce_sum + bce.item(), aux_sum + out.aux_loss.item(), count + 1
            if step % config.frame_every == 0 or step == config.steps:
                self._record(history, evaluator, ema.module, step, bce_sum / count,
                             aux_sum / count)
                bce_sum, aux_sum, count = 0.0, 0.0, 0
        return ema.module

    def _record(self, history, evaluator, model, step, batch_bce, aux_loss):
        """Evaluates, logs, keeps the frame state (when filming) and prints now and then."""
        evaluation = evaluator.evaluate(model)
        metrics, loads = evaluation.metrics, evaluation.loads
        history.log(step, {"train BCE": metrics["train BCE"], "test BCE": metrics["test BCE"]},
                    evaluation.scores.by_kind(), loads,
                    {"batch BCE": batch_bce, "aux loss": aux_loss})
        if not self.config.no_video:
            history.keep(step, evaluation.state)
        if len(history.steps) % PRINT_EVERY_FRAMES == 1 or step == self.config.steps:
            print(f"step {step:5d}  train BCE {metrics['train BCE']:.4f}  test BCE "
                  f"{metrics['test BCE']:.4f}  errors {metrics['errors']:3d}  load "
                  + " | ".join(" ".join(f"{v:.2f}" for v in load) for load in loads), flush=True)
