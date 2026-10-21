"""Splitting text into tokens, BERT-style but with whole words where BERT uses word pieces."""

import re
from dataclasses import dataclass
from typing import ClassVar

SPECIALS = ("[PAD]", "[UNK]", "[CLS]")
PAD, UNK, CLS = range(len(SPECIALS))
NUMBER_TOKENS = ("[NUM]", "[NUM5+]")


@dataclass(frozen=True)
class Tokenizer:
    """Turns text into lowercase words, numbers and single punctuation marks.

    Digit runs become ``[NUM]`` (shorter than ``long_number`` digits) or ``[NUM5+]`` (phone
    numbers and short codes), so they share one common token instead of each being rare.

    Attributes:
        long_number: The number of digits from which a number counts as a phone number or
            short code.
    """

    long_number: int = 5
    PIECE: ClassVar[re.Pattern] = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\sa-z\d]")

    def pieces(self, text):
        """Returns the lowercased words, digit runs and punctuation marks of ``text``."""
        return self.PIECE.findall(text.lower())

    def tokenize(self, text):
        """Returns the tokens of ``text``: its pieces, with every digit run a number token."""
        return [self._token(piece) for piece in self.pieces(text)]

    def _token(self, piece):
        if piece.isdecimal():
            return NUMBER_TOKENS[len(piece) >= self.long_number]
        return piece

    @staticmethod
    def is_number(token):
        """Returns whether ``token`` is one of the number tokens."""
        return token in NUMBER_TOKENS

    @staticmethod
    def is_punctuation(token):
        """Returns whether ``token`` is a punctuation mark or symbol.

        A token is punctuation when it is not a special token, not a number token and does
        not start with a letter.
        """
        return (token not in SPECIALS and token not in NUMBER_TOKENS
                and not token[0].isalpha())


TOKENIZER = Tokenizer()
