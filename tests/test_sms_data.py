import hashlib

import pytest

from moe_spam.data.dataset import MAX_LEN, SpamData
from moe_spam.data.sms import Message, SmsSpamCollection
from moe_spam.errors import MoeSpamError
from moe_spam.paths import PATHS


def write_data(path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_parses_labels_and_keeps_tabs_in_text(tmp_path):
    path = tmp_path / "sms"
    sha = write_data(path, ["ham\tsee you\tlater", "spam\tWIN now"])
    assert SmsSpamCollection(path, sha).load() == [Message(False, "see you\tlater"),
                                                   Message(True, "WIN now")]


def test_load_refuses_a_file_with_the_wrong_checksum(tmp_path):
    path = tmp_path / "sms"
    write_data(path, ["ham\thi"])
    with pytest.raises(MoeSpamError, match="checksum"):
        SmsSpamCollection(path, "0" * 64).load()


def test_load_explains_how_to_get_a_missing_file(tmp_path):
    with pytest.raises(MoeSpamError, match="repository root"):
        SmsSpamCollection(tmp_path / "missing", "0" * 64).load()


def test_load_rejects_unknown_labels(tmp_path):
    path = tmp_path / "sms"
    sha = write_data(path, ["ham\thi", "eggs\tbad line"])
    with pytest.raises(MoeSpamError, match="line 2"):
        SmsSpamCollection(path, sha).load()


def test_normalise_ignores_case_spacing_and_digits():
    normalise = SmsSpamCollection.normalise
    assert normalise("Call  09061 NOW\n") == normalise("call 12345 now") == "call 0 now"


def test_dedupe_keeps_the_first_of_each_normalised_text():
    messages = [Message(True, "Win 100 now"), Message(False, "hi"), Message(True, "win 250  NOW")]
    assert SmsSpamCollection.dedupe(messages) == [Message(True, "Win 100 now"),
                                                  Message(False, "hi")]


def test_split_is_stratified_seeded_and_disjoint():
    messages = [Message(i % 5 == 0, f"message {chr(97 + i % 26)} {i // 26}") for i in range(200)]
    split = SmsSpamCollection.split
    train, test = split(messages, test_fraction=0.25, seed=3)
    assert len(test) == 50 and len(train) == 150
    assert sum(m.is_spam for m in test) == 10  # 25% of the 40 spam
    assert not set(train) & set(test)
    assert split(messages, test_fraction=0.25, seed=3) == (train, test)
    assert split(messages, test_fraction=0.25, seed=4) != (train, test)


def test_split_rejects_fractions_outside_zero_one():
    with pytest.raises(ValueError, match="test_fraction"):
        SmsSpamCollection.split([Message(True, "a")], test_fraction=1.0)


def test_prepare_learns_only_from_the_training_half():
    messages = [Message(i % 4 == 0, f"{'win cash' if i % 4 == 0 else 'see you'} {chr(97 + i)}")
                for i in range(20)]
    data = SpamData.prepare(messages)
    assert len(data.train_ids) + len(data.test_ids) == 20
    assert data.test_ids.shape[1] == MAX_LEN
    assert data.test_labels.tolist() == [float(m.is_spam) for m in data.test_messages]
    assert data.n_spam + data.n_ham == len(data.test_ids)
    train_texts = {m.text for m in messages} - {m.text for m in data.test_messages}
    assert set(data.vocab.words[3:]) <= {t for text in train_texts for t in text.split()}


@pytest.mark.skipif(not PATHS.data_file.is_file(), reason="dataset not downloaded")
def test_real_dataset_counts():
    collection = SmsSpamCollection()
    messages = collection.load()
    assert len(messages) == 5574
    assert sum(m.is_spam for m in messages) == 747
    unique = collection.dedupe(messages)
    assert len(unique) == 5137
    train, test = collection.split(unique)
    normalise = collection.normalise
    assert {normalise(m.text) for m in train}.isdisjoint(normalise(m.text) for m in test)
