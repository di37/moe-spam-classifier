import random

import pytest

from moe_spam.analysis.benchmark import Benchmark
from moe_spam.data.sms import Message, SmsSpamCollection

HAM_WORDS = ("see", "you", "at", "lunch", "tomorrow", "ok", "thanks", "call", "me", "later",
             "home", "now")
SPAM_WORDS = ("win", "free", "prize", "cash", "claim", "txt", "now", "urgent", "call", "stop")


def make_messages(n=240, seed=0):
    """Ham and spam built from two small word lists; spam ends with a 5-digit short code."""
    rng = random.Random(seed)
    messages = []
    for i in range(n):
        is_spam = i % 4 == 0
        pool = SPAM_WORDS if is_spam else HAM_WORDS
        words = [rng.choice(pool) for _ in range(rng.randint(8, 12))]
        tail = f" {rng.randint(10000, 99999)}" if is_spam else ""
        messages.append(Message(is_spam, " ".join(words) + tail))
    return messages


@pytest.fixture
def synthetic_messages():
    """240 synthetic messages, a quarter of them spam."""
    return make_messages()


@pytest.fixture
def tiny_project(tmp_path, monkeypatch):
    """Run in an empty folder, on synthetic messages instead of the real dataset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SmsSpamCollection, "load", lambda self: make_messages())
    return tmp_path


@pytest.fixture
def fast_benchmark():
    """A benchmark small and short enough for tests."""
    return Benchmark(sizes={"ours": (16, 32, 3), "large": (32, 64, 4)}, batch_tokens=64, reps=2,
                     warmup=1, latency_tokens=4, latency_reps=2, latency_warmup=1)


@pytest.fixture
def measured():
    """A benchmark record, as ``Benchmark.record`` saves it."""
    return {
        "sizes": {"ours": {"d_model": 64, "d_ff": 256, "n_experts": 3, "batch_tokens": 8192,
                           "latency_tokens": 24,
                           "ms_per_1k_tokens": {"dense": 0.1, "wide": 0.4, "moe": 0.42},
                           "latency_us": {"dense": 20.0, "wide": 56.0, "moe": 1400.0}},
                  "large": {"d_model": 1024, "d_ff": 4096, "n_experts": 8, "batch_tokens": 8192,
                            "latency_tokens": 24,
                            "ms_per_1k_tokens": {"dense": 7.0, "wide": 60.0, "moe": 11.0},
                            "latency_us": {"dense": 1200.0, "wide": 4700.0, "moe": 12000.0}}},
        "models": {"dense": {"one_message_ms": 1.4, "ms_per_1k_messages": 85.0},
                   "moe": {"one_message_ms": 3.0, "ms_per_1k_messages": 97.0}},
        "test_messages": 1027,
    }
