"""Running a model without dropout, then giving it back its mode."""

from contextlib import contextmanager


@contextmanager
def evaluating(model):
    """Puts ``model`` in eval mode inside the ``with`` block, and its own mode back after it.

    Args:
        model: Any ``torch.nn.Module``.

    Yields:
        The model, in eval mode.
    """
    was_training = model.training
    model.eval()
    try:
        yield model
    finally:
        model.train(was_training)
