"""
How hard is the task? A bag-of-words logistic regression on the same split and vocabulary,
next to every transformer run recorded in results/.

    python baselines.py

Writes results/baselines.json and prints one table with all of them.
"""
import glob
import os

import torch
import torch.nn.functional as F

from dataset import prepare_data
from run_log import RESULTS_DIR, read_json, write_json
from vocab import PAD

STEPS = 500
LR = 0.05
WEIGHT_DECAY = 1e-4
SEED = 0
COLUMNS = ("accuracy", "errors", "spam missed", "ham flagged", "spam F1")


def bag_of_words(ids, vocab_size):
    """[n, vocab_size]: 1 where the message contains that token, 0 elsewhere ([PAD] is never 1)."""
    present = torch.zeros(len(ids), vocab_size).scatter(1, ids, 1.0)
    return present * (torch.arange(vocab_size) != PAD)


def fit_logistic_regression(features, labels, steps=STEPS, seed=SEED):
    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(steps):  # the whole training set in every step: it is small
        loss = F.binary_cross_entropy_with_logits(model(features).squeeze(-1), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def score(logits, labels):
    """The same numbers train_moe.py reports, for logits > 0 meaning spam."""
    is_spam = labels.bool()
    wrong = (logits > 0) != is_spam
    missed, flagged = int((wrong & is_spam).sum()), int((wrong & ~is_spam).sum())
    caught = int(is_spam.sum()) - missed
    return {"accuracy": 1 - int(wrong.sum()) / len(labels), "errors": int(wrong.sum()),
            "spam missed": missed, "ham flagged": flagged,
            "spam F1": 2 * caught / (2 * caught + missed + flagged)}


def results_table(runs):
    """runs: {name: metrics}. One aligned row per run."""
    header = f"{'run':<22}" + "".join(f"{c:>13}" for c in COLUMNS)
    rows = [f"{name:<22}{m['accuracy']:>12.1%} {m['errors']:>12} {m['spam missed']:>12} "
            f"{m['ham flagged']:>12} {m['spam F1']:>12.3f}" for name, m in runs.items()]
    return "\n".join([header, *rows])


def main():
    data = prepare_data()
    model = fit_logistic_regression(bag_of_words(data.train_ids, len(data.vocab)),
                                     data.train_labels)
    with torch.no_grad():
        logits = model(bag_of_words(data.test_ids, len(data.vocab))).squeeze(-1)
    bow = score(logits, data.test_labels)
    write_json(os.path.join(RESULTS_DIR, "baselines.json"),
               {"bag-of-words logistic regression": bow, "steps": STEPS, "lr": LR,
                "weight_decay": WEIGHT_DECAY, "seed": SEED})
    runs = {"bag-of-words logreg": bow}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "training_*.json"))):
        name = os.path.basename(path).removeprefix("training_").removesuffix(".json")
        runs[name] = read_json(path)["final_metrics"]
    print(results_table(runs))


if __name__ == "__main__":
    main()
