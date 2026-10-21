import hashlib
import os

import pytest

from sms_data import DATA_PATH, Message, dedupe, load_messages, normalise, split


def write_data(path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_messages_parses_labels_and_keeps_tabs_in_text(tmp_path):
    path = tmp_path / "sms"
    sha = write_data(path, ["ham\tsee you\tlater", "spam\tWIN now"])
    assert load_messages(str(path), sha) == [Message(False, "see you\tlater"),
                                             Message(True, "WIN now")]


def test_load_messages_refuses_a_file_with_the_wrong_checksum(tmp_path):
    path = tmp_path / "sms"
    write_data(path, ["ham\thi"])
    with pytest.raises(SystemExit, match="checksum"):
        load_messages(str(path), "0" * 64)


def test_load_messages_explains_how_to_get_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="download"):
        load_messages(str(tmp_path / "missing"), "0" * 64)


def test_load_messages_rejects_unknown_labels(tmp_path):
    path = tmp_path / "sms"
    sha = write_data(path, ["ham\thi", "eggs\tbad line"])
    with pytest.raises(ValueError, match="line 2"):
        load_messages(str(path), sha)


def test_normalise_ignores_case_spacing_and_digits():
    assert normalise("Call  09061 NOW\n") == normalise("call 12345 now") == "call 0 now"


def test_dedupe_keeps_the_first_of_each_normalised_text():
    messages = [Message(True, "Win 100 now"), Message(False, "hi"), Message(True, "win 250  NOW")]
    assert dedupe(messages) == [Message(True, "Win 100 now"), Message(False, "hi")]


def test_split_is_stratified_seeded_and_disjoint():
    messages = [Message(i % 5 == 0, f"message {chr(97 + i % 26)} {i // 26}") for i in range(200)]
    train, test = split(messages, test_fraction=0.25, seed=3)
    assert len(test) == 50 and len(train) == 150
    assert sum(m.is_spam for m in test) == 10  # 25% of the 40 spam
    assert not set(train) & set(test)
    assert split(messages, test_fraction=0.25, seed=3) == (train, test)
    assert split(messages, test_fraction=0.25, seed=4) != (train, test)


def test_split_rejects_fractions_outside_zero_one():
    with pytest.raises(ValueError, match="test_fraction"):
        split([Message(True, "a")], test_fraction=1.0)


@pytest.mark.skipif(not os.path.isfile(DATA_PATH), reason="dataset not downloaded")
def test_real_dataset_counts():
    messages = load_messages()
    assert len(messages) == 5574
    assert sum(m.is_spam for m in messages) == 747
    unique = dedupe(messages)
    assert len(unique) == 5137
    train, test = split(unique)
    assert {normalise(m.text) for m in train}.isdisjoint(normalise(m.text) for m in test)
