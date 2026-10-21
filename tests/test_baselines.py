import torch

from baselines import bag_of_words, fit_logistic_regression, results_table, score
from vocab import CLS, PAD


def test_bag_of_words_marks_present_tokens_and_ignores_padding():
    ids = torch.tensor([[CLS, 5, 5, 3, PAD], [CLS, 4, PAD, PAD, PAD]])
    bow = bag_of_words(ids, vocab_size=6)
    assert bow.tolist() == [[0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 0]]


def test_logistic_regression_learns_a_separable_toy_task():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    model = fit_logistic_regression(features, labels, steps=300)
    with torch.no_grad():
        metrics = score(model(features).squeeze(-1), labels)
    assert metrics["errors"] == 0
    assert metrics["spam F1"] == 1.0


def test_score_counts_both_kinds_of_mistake():
    logits = torch.tensor([3.0, -2.0, -1.0, 2.0])  # says spam, ham, ham, spam
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    metrics = score(logits, labels)
    assert (metrics["spam missed"], metrics["ham flagged"], metrics["errors"]) == (1, 1, 2)
    assert metrics["accuracy"] == 0.5


def test_results_table_lists_every_run():
    runs = {"moe": {"accuracy": 0.98, "errors": 17, "spam missed": 12, "ham flagged": 5,
                    "spam F1": 0.93}}
    table = results_table(runs)
    assert table.splitlines()[0].startswith("run")
    assert "moe" in table and "98.3%" not in table and "98.0%" in table
