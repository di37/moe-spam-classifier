import numpy as np
import pytest

from moe_spam.errors import MoeSpamError
from moe_spam.records import RecordStore
from moe_spam.training import baselines
from moe_spam.training.runs import (
    RunHistory,
    RunSummary,
    load_baseline,
    load_history,
    load_summary,
)

RECORD = {
    "args": {"experts": 3, "aux_weight": 0.0, "seed": 2, "steps": 10},
    "final_metrics": {"accuracy": 0.98, "errors": 17, "spam F1": 0.93,
                      "layer 1 load": [0.5, 0.3, 0.2], "layer 2 load": [0.1, 0.8, 0.1]},
}
ROWS = [{"step": 0, "aux loss": "nan", "layer 1 load expert 1": 0.4, "layer 1 load expert 2": 0.3,
         "layer 1 load expert 3": 0.3, "layer 2 load expert 1": 0.3,
         "layer 2 load expert 2": 0.4, "layer 2 load expert 3": 0.3, "test BCE": 0.7},
        {"step": 5, "aux loss": 1.4, "layer 1 load expert 1": 0.5, "layer 1 load expert 2": 0.3,
         "layer 1 load expert 3": 0.2, "layer 2 load expert 1": 0.1,
         "layer 2 load expert 2": 0.8, "layer 2 load expert 3": 0.1, "test BCE": 0.1}]


@pytest.fixture
def store(tmp_path):
    store = RecordStore(tmp_path)
    store.write_json("training_moe_noaux_seed2", RECORD)
    store.write_csv("training_moe_noaux_seed2", ROWS)
    return store


def test_summary_reads_the_settings_and_final_numbers(store):
    run = load_summary("moe_noaux_seed2", store)
    assert run == RunSummary("moe_noaux_seed2", experts=3, aux_weight=0.0, seed=2,
                             accuracy=0.98, errors=17, spam_f1=0.93,
                             loads=((0.5, 0.3, 0.2), (0.1, 0.8, 0.1)))
    assert run.busiest(1) == 0.5 and run.busiest(2) == 0.8


def test_history_reads_every_logged_step(store):
    history = load_history("moe_noaux_seed2", store)
    assert isinstance(history, RunHistory)
    assert history.steps.tolist() == [0, 5]
    assert history.n_experts == 3 and len(history.loads) == 2
    assert np.allclose(history.loads[1][1], [0.1, 0.8, 0.1])
    assert np.isnan(history.aux[0]) and history.aux[1] == pytest.approx(1.4)
    assert history.busiest(2).tolist() == pytest.approx([0.4, 0.8])


def test_baseline_is_read_from_its_record(store):
    with pytest.raises(MoeSpamError, match="moe-spam baselines"):
        load_baseline(store)
    scores = {"errors": 16, "accuracy": 0.984}
    store.write_json(baselines.RECORD, {baselines.RECORD_KEY: scores, "lr": 0.05})
    assert load_baseline(store) == {"errors": 16, "accuracy": 0.984}


def test_missing_runs_say_how_to_make_them(store):
    with pytest.raises(MoeSpamError, match="moe-spam train"):
        load_summary("moe_experts9", store)
    with pytest.raises(MoeSpamError, match="moe-spam train"):
        load_history("moe_experts9", store)
