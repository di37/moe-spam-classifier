"""
The SMS Spam Collection (Almeida, Gómez Hidalgo and Yamakami, 2011): 5,574 text messages,
each labelled ham (a normal message) or spam.

  load_messages   read the file and check it is the one this project was built on
  normalise       lowercase, single spaces, every digit run -> 0
  dedupe          drop messages whose normalised text repeats an earlier one
  split           seeded train/test split that keeps the ham/spam ratio in both halves
"""
import hashlib
import os
import random
import re
from typing import NamedTuple

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "raw", "sms", "SMSSpamCollection")
DATA_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
LABELS = {"ham": False, "spam": True}


class Message(NamedTuple):
    is_spam: bool
    text: str


def _parse_line(path, number, line):
    label, _, text = line.partition("\t")
    if label not in LABELS:
        raise ValueError(f"{path} line {number}: unknown label {label!r}")
    return Message(LABELS[label], text)


def load_messages(path=DATA_PATH, sha256=DATA_SHA256):
    """One Message per line of the tab-separated file `label<TAB>text`."""
    if not os.path.isfile(path):
        raise SystemExit(f"{path} not found: download {DATA_URL} and unzip it into "
                         f"{os.path.dirname(path) or '.'}/ (see README)")
    with open(path, "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise SystemExit(f"{path} does not match the expected checksum; "
                         f"download it again from {DATA_URL}")
    lines = raw.decode("utf-8").splitlines()
    return [_parse_line(path, number, line) for number, line in enumerate(lines, start=1)]


def normalise(text):
    return re.sub(r"\d+", "0", " ".join(text.lower().split()))


def dedupe(messages):
    """
    Many spams are one template sent with different numbers or prize amounts. Comparing
    normalised texts keeps those copies from landing on both sides of the train/test split.
    """
    first = {}
    for message in messages:
        first.setdefault(normalise(message.text), message)
    return list(first.values())


def split(messages, test_fraction=0.2, seed=0):
    """Returns (train, test), each in a seeded random order."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}")
    rng = random.Random(seed)
    train, test = [], []
    for is_spam in (False, True):
        group = [m for m in messages if m.is_spam == is_spam]
        shuffled = rng.sample(group, len(group))
        n_test = round(len(group) * test_fraction)
        test = test + shuffled[:n_test]
        train = train + shuffled[n_test:]
    return rng.sample(train, len(train)), rng.sample(test, len(test))
