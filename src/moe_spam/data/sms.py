"""The SMS Spam Collection: 5,574 text messages, each labelled ham (normal) or spam.

Almeida, Gómez Hidalgo and Yamakami, "Contributions to the Study of SMS Spam Filtering: New
Collection and Results", ACM DocEng 2011. Distributed by the UCI Machine Learning Repository.
"""

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from moe_spam.errors import MoeSpamError
from moe_spam.paths import PATHS

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_SHA256 = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
LABELS = {"ham": False, "spam": True}


class Message(NamedTuple):
    """One labelled text message.

    Attributes:
        is_spam: Whether the message is spam.
        text: The message as typed.
    """

    is_spam: bool
    text: str


@dataclass(frozen=True)
class SmsSpamCollection:
    """The dataset file, and the clean-up and split that every run uses.

    Attributes:
        path: Where the unzipped file is.
        sha256: The checksum the file must have.
    """

    path: Path = PATHS.data_file
    sha256: str = DATA_SHA256

    def load(self):
        """Reads the file and checks it is the one this project was built on.

        Returns:
            One ``Message`` per line of the tab-separated file ``label<TAB>text``.

        Raises:
            MoeSpamError: If the file is missing, its checksum is wrong, or a line has an
                unknown label.
        """
        path = Path(self.path)
        if not path.is_file():
            raise MoeSpamError(f"{path} not found: run from the repository root, or download "
                             f"{DATA_URL} and unzip it into {path.parent}/ (see README)")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise MoeSpamError(f"{path} does not match the expected checksum; "
                             f"download it again from {DATA_URL}")
        lines = raw.decode("utf-8").splitlines()
        return [self._parse(path, number, line) for number, line in enumerate(lines, start=1)]

    @staticmethod
    def _parse(path, number, line):
        label, _, text = line.partition("\t")
        if label not in LABELS:
            raise MoeSpamError(f"{path} line {number}: unknown label {label!r}")
        return Message(LABELS[label], text)

    @staticmethod
    def normalise(text):
        """Returns ``text`` lowercased, with single spaces and every digit run replaced by 0."""
        return re.sub(r"\d+", "0", " ".join(text.lower().split()))

    @classmethod
    def dedupe(cls, messages):
        """Drops every message whose normalised text repeats an earlier one.

        Many spams are one template sent with different numbers or prize amounts. Comparing
        normalised texts keeps those copies from landing on both sides of the split.

        Args:
            messages: The messages, in order.

        Returns:
            The first message of each normalised text, in the original order.
        """
        first = {}
        for message in messages:
            first.setdefault(cls.normalise(message.text), message)
        return list(first.values())

    @staticmethod
    def split(messages, test_fraction=0.2, seed=0):
        """Splits messages into training and test sets, keeping the ham/spam ratio in both.

        Args:
            messages: The messages to split.
            test_fraction: The share of each label that goes to the test set.
            seed: Seed of the shuffle.

        Returns:
            ``(train, test)``, each in a seeded random order.

        Raises:
            ValueError: If ``test_fraction`` is not strictly between 0 and 1.
        """
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
