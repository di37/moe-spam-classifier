"""How well a classifier separates spam from ham, from its mistakes."""

from dataclasses import dataclass

SPAM_MISSED, HAM_FLAGGED = "spam missed", "ham flagged"
COLUMNS = ("accuracy", "errors", SPAM_MISSED, HAM_FLAGGED, "spam F1")  # the keys of as_dict()


@dataclass(frozen=True)
class SpamScores:
    """The mistakes on a test set, and the scores that follow from them.

    Attributes:
        n: Test messages.
        n_spam: Spam among them.
        spam_missed: Spam the classifier let through.
        ham_flagged: Ham the classifier marked as spam.
    """

    n: int
    n_spam: int
    spam_missed: int
    ham_flagged: int

    @classmethod
    def from_logits(cls, logits, labels):
        """Scores logits (above 0 means spam) against labels (1 for spam)."""
        is_spam = labels.bool()
        wrong = (logits > 0) != is_spam
        return cls(len(labels), int(is_spam.sum()), int((wrong & is_spam).sum()),
                   int((wrong & ~is_spam).sum()))

    @property
    def errors(self):
        """All mistakes."""
        return self.spam_missed + self.ham_flagged

    @property
    def accuracy(self):
        """The share of messages classified right."""
        return 1 - self.errors / self.n

    @property
    def spam_f1(self):
        """The F1 score of the spam class."""
        caught = self.n_spam - self.spam_missed
        return 2 * caught / (2 * caught + self.ham_flagged + self.spam_missed)

    def by_kind(self):
        """Returns the two kinds of mistake under the names the records use."""
        return {SPAM_MISSED: self.spam_missed, HAM_FLAGGED: self.ham_flagged}

    def as_dict(self):
        """Returns the scores under the ``COLUMNS`` names the records and tables use."""
        return {"accuracy": self.accuracy, "errors": self.errors, **self.by_kind(),
                "spam F1": self.spam_f1}
