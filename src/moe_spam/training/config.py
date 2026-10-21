"""The settings of a training run: the one place their defaults are written down."""

import math
from dataclasses import asdict, dataclass

from moe_spam.model.bert import N_EXPERTS

EMA_DECAY = 0.99      # frames, checkpoint and every reported number use this moving average
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
VIDEO_EXPERTS = N_EXPERTS  # the training video's routing plot is a triangle: one corner each
NAMED_SETTINGS = ("experts", "aux_weight", "steps", "batch_size", "lr", "seed")


@dataclass(frozen=True)
class TrainingConfig:
    """What a training run does.

    Attributes:
        steps: Training steps, one batch each.
        batch_size: Messages per step.
        lr: Peak AdamW learning rate.
        experts: Experts per MoE layer (1 makes a plain dense BERT).
        aux_weight: Weight of the load-balancing loss (0 switches it off).
        frame_every: Evaluate, and keep a frame, every this many steps (and at the last).
        seed: Seed of the weights and the batches; the train/test split never changes.
        no_video: Only train and write the run record, without frames or video.
    """

    steps: int = 800
    batch_size: int = 64
    lr: float = 1e-3
    experts: int = N_EXPERTS
    aux_weight: float = 0.01
    frame_every: int = 5
    seed: int = 0
    no_video: bool = False

    def __post_init__(self):
        """Checks the settings.

        Raises:
            ValueError: If a count is below 1, ``lr`` is not a positive number,
                ``aux_weight`` is negative or not a number, or a video is asked for with a
                number of experts other than ``VIDEO_EXPERTS``.
        """
        for name in ("steps", "batch_size", "experts", "frame_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(f"lr must be a number more than 0, got {self.lr}")
        if not math.isfinite(self.aux_weight) or self.aux_weight < 0:
            raise ValueError(f"aux_weight must be a number at least 0, got {self.aux_weight}")
        if self.experts != VIDEO_EXPERTS and not self.no_video:
            raise ValueError(f"the video's routing plot needs {VIDEO_EXPERTS} experts; "
                             f"train with no_video to use {self.experts}")

    @property
    def run_name(self):
        """``moe``, ``moe_noaux`` or ``dense``, then every setting that differs from its default.

        For example ``moe_steps400_seed2``. Settings that do not change training, like
        ``frame_every``, are left out, so different runs never share files.
        """
        base = "dense" if self.experts == 1 else ("moe_noaux" if self.aux_weight == 0 else "moe")
        defaults = TrainingConfig()
        in_base = {"experts": (1, defaults.experts), "aux_weight": (0, defaults.aux_weight)}
        changed = [f"{key.replace('_', '')}{getattr(self, key)}" for key in NAMED_SETTINGS
                   if getattr(self, key) != getattr(defaults, key)
                   and getattr(self, key) not in in_base.get(key, ())]
        return "_".join([base, *changed])

    def as_record(self):
        """Returns the settings as a dict, for the run record."""
        return asdict(self)


MOE_RUN = TrainingConfig().run_name                           # "moe": the model the videos draw
DENSE_RUN = TrainingConfig(experts=1, no_video=True).run_name  # "dense": the one-expert model
