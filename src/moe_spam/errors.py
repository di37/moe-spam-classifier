"""The one error type for problems the user can fix."""


class MoeSpamError(Exception):
    """A missing or wrong file, an unsuitable checkpoint, or data that does not fit a step.

    The ``moe-spam`` command prints its message and exits with status 1, without a
    traceback. Anything else that goes wrong is a bug, and keeps its traceback.
    """
