import torch

from moe_spam.data.tokenizer import CLS, PAD
from moe_spam.training.baselines import BagOfWordsBaseline, results_table
from moe_spam.training.scores import COLUMNS, SpamScores


def test_bag_of_words_marks_present_tokens_and_ignores_padding():
    ids = torch.tensor([[CLS, 5, 5, 3, PAD], [CLS, 4, PAD, PAD, PAD]])
    bow = BagOfWordsBaseline.features(ids, vocab_size=6)
    assert bow.tolist() == [[0, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 0]]


def test_logistic_regression_learns_a_separable_toy_task():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    model = BagOfWordsBaseline(steps=300).fit(features, labels)
    with torch.no_grad():
        scores = SpamScores.from_logits(model(features).squeeze(-1), labels)
    assert scores.errors == 0
    assert scores.spam_f1 == 1.0


def test_scores_count_both_kinds_of_mistake():
    logits = torch.tensor([3.0, -2.0, -1.0, 2.0])  # says spam, ham, ham, spam
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    scores = SpamScores.from_logits(logits, labels)
    assert (scores.spam_missed, scores.ham_flagged, scores.errors) == (1, 1, 2)
    assert scores.accuracy == 0.5
    assert scores.as_dict()["spam F1"] == scores.spam_f1
    assert tuple(scores.as_dict()) == COLUMNS
    assert scores.by_kind() == {"spam missed": 1, "ham flagged": 1}


def test_results_table_lists_every_run():
    runs = {"moe": {"accuracy": 0.98, "errors": 17, "spam missed": 12, "ham flagged": 5,
                    "spam F1": 0.93}}
    table = results_table(runs)
    assert table.splitlines()[0].startswith("run")
    assert "moe" in table and "98.3%" not in table and "98.0%" in table
    assert table.splitlines()[1] == (f"{'moe':<22}{0.98:>12.1%} {17:>12} {12:>12} {5:>12} "
                                     f"{0.93:>12.3f}")


def test_baseline_records_its_settings():
    assert BagOfWordsBaseline().as_record() == {"steps": 500, "lr": 0.05, "weight_decay": 1e-4,
                                                "seed": 0}
