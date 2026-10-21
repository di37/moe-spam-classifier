"""The vocabulary (token ↔ id) and the kind of each token, for the training video's heatmap."""

from collections import Counter
from dataclasses import dataclass, field

import torch

from moe_spam.data.tokenizer import CLS, PAD, SPECIALS, TOKENIZER, UNK

MAX_LEN = 64  # [CLS] + 63 tokens holds 98.8% of the messages whole

GROUPS = ("[CLS]", "spam-leaning words", "ham-leaning words", "other words", "numbers",
          "punctuation", "[UNK]")


@dataclass(frozen=True)
class Vocab:
    """Maps tokens to ids and back.

    Attributes:
        words: Every token, in id order: the special tokens first.
    """

    words: tuple
    _index: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Builds the token → id index."""
        object.__setattr__(self, "_index", {w: i for i, w in enumerate(self.words)})

    @classmethod
    def build(cls, texts, min_count=2, tokenizer=TOKENIZER):
        """Builds a vocabulary from training texts.

        Args:
            texts: The training texts.
            min_count: How often a token must appear to get its own id.
            tokenizer: How the texts are split into tokens.

        Returns:
            The special tokens, then every token seen at least ``min_count`` times, most
            frequent first (ties alphabetically).

        Raises:
            ValueError: If ``min_count`` is below 1.
        """
        if min_count < 1:
            raise ValueError(f"min_count must be at least 1, got {min_count}")
        counts = Counter(t for text in texts for t in tokenizer.tokenize(text))
        kept = sorted((w for w, c in counts.items() if c >= min_count),
                      key=lambda w: (-counts[w], w))
        return cls(SPECIALS + tuple(kept))

    def __len__(self):
        """Returns the number of tokens."""
        return len(self.words)

    def id(self, token):
        """Returns the id of ``token``, or the ``[UNK]`` id for a token not in the vocabulary."""
        return self._index.get(token, UNK)

    def knows(self, tokens):
        """Returns whether every one of ``tokens`` has its own id (none would be ``[UNK]``)."""
        return all(self.id(t) != UNK for t in tokens)

    def encode(self, texts, max_len=MAX_LEN, tokenizer=TOKENIZER):
        """Encodes texts as rows of token ids.

        Args:
            texts: The texts to encode.
            max_len: The length of every row, ``[CLS]`` included.
            tokenizer: How the texts are split into tokens.

        Returns:
            A LongTensor ``[len(texts), max_len]``: ``[CLS]``, the tokens (cut to fit), then
            ``[PAD]``s.

        Raises:
            ValueError: If ``max_len`` leaves no room for a token after ``[CLS]``.
        """
        if max_len < 2:
            raise ValueError(f"max_len must be at least 2 ([CLS] and one token), got {max_len}")
        rows = [[CLS] + [self.id(t) for t in tokenizer.tokenize(text)][: max_len - 1]
                for text in texts]
        padded = [row + [PAD] * (max_len - len(row)) for row in rows]
        return torch.tensor(padded, dtype=torch.long).reshape(len(texts), max_len)


@dataclass(frozen=True)
class TokenGroups:
    """The kind of every token in a vocabulary: spam-leaning words, numbers, punctuation, ...

    Attributes:
        ids: LongTensor ``[len(vocab)]``: each token's index into ``GROUPS``, -1 for ``[PAD]``.
    """

    ids: torch.Tensor

    @classmethod
    def from_messages(cls, vocab, messages, min_count=5, spam_leaning=0.8, ham_leaning=0.02,
                      tokenizer=TOKENIZER):
        """Sorts a vocabulary's tokens into groups, by how they are used in ``messages``.

        A word is spam-leaning if at least ``spam_leaning`` of its uses are in spam, and
        ham-leaning if at most ``ham_leaning`` are. Words used fewer than ``min_count`` times
        are "other words". Pass the training messages, so the test set stays unseen.

        Args:
            vocab: The vocabulary to sort.
            messages: Labelled messages (``is_spam``, ``text``).
            min_count: Uses a word needs before it can lean either way.
            spam_leaning: Share of spam uses from which a word is spam-leaning.
            ham_leaning: Share of spam uses up to which a word is ham-leaning.
            tokenizer: How the messages are split into tokens.

        Returns:
            The groups of every token in ``vocab``.
        """
        total = Counter(t for m in messages for t in tokenizer.tokenize(m.text))
        spam = Counter(t for m in messages if m.is_spam for t in tokenizer.tokenize(m.text))

        def group(token):
            if token == SPECIALS[PAD]:
                return -1
            if token in (SPECIALS[CLS], SPECIALS[UNK]):
                return GROUPS.index(token)
            if tokenizer.is_number(token):
                return GROUPS.index("numbers")
            if tokenizer.is_punctuation(token):
                return GROUPS.index("punctuation")
            uses = total[token]
            if uses >= min_count and spam[token] / uses >= spam_leaning:
                return GROUPS.index("spam-leaning words")
            if uses >= min_count and spam[token] / uses <= ham_leaning:
                return GROUPS.index("ham-leaning words")
            return GROUPS.index("other words")

        return cls(torch.tensor([group(t) for t in vocab.words], dtype=torch.long))

    def of(self, token_ids):
        """Returns the group of every token id in ``token_ids`` (any shape)."""
        return self.ids[token_ids]

    def labels(self, token_ids):
        """Returns each group's name with how many of ``token_ids`` (pads aside) it has."""
        counts = torch.bincount(self.of(token_ids[token_ids != PAD]), minlength=len(GROUPS))
        return [f"{name} ({count:,})" for name, count in zip(GROUPS, counts.tolist(), strict=True)]

    @staticmethod
    def expert_shares(groups, experts, n_experts):
        """Returns how each group's tokens are shared out among the experts.

        Args:
            groups: The group of each token (no pads).
            experts: The expert that processed each token.
            n_experts: The number of experts.

        Returns:
            Array ``[len(GROUPS), n_experts]``: of each group's tokens, the share each expert
            ran. Rows of groups with no tokens are NaN.
        """
        n_groups = len(GROUPS)
        counts = torch.bincount(groups * n_experts + experts, minlength=n_groups * n_experts)
        counts = counts.reshape(n_groups, n_experts).double()
        return (counts / counts.sum(dim=1, keepdim=True)).numpy()
