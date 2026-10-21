import pytest
import torch

from sms_data import Message
from vocab import CLS, GROUPS, PAD, UNK, Vocab, encode, pieces, token_groups, tokenize


def test_tokenize_splits_words_numbers_and_punctuation():
    assert tokenize("WINNER!! Call 09061701461 now, 4 a £900 prize") == [
        "winner", "!", "!", "call", "[NUM5+]", "now", ",", "[NUM]", "a", "£", "[NUM]", "prize"]


def test_tokenize_keeps_apostrophes_inside_words():
    assert tokenize("I'd say it's ok") == ["i'd", "say", "it's", "ok"]


def test_pieces_are_the_lowercased_surface_forms_of_the_tokens():
    text = "Txt 87121 NOW!"
    assert pieces(text) == ["txt", "87121", "now", "!"]
    assert len(pieces(text)) == len(tokenize(text))


def test_vocab_puts_specials_first_then_words_by_frequency():
    vocab = Vocab.build(["b a a", "c b a", "c"], min_count=1)
    assert vocab.words[:3] == ("[PAD]", "[UNK]", "[CLS]")
    assert vocab.words[3:] == ("a", "b", "c")  # ties broken alphabetically
    assert (vocab.id("[PAD]"), vocab.id("[UNK]"), vocab.id("[CLS]")) == (PAD, UNK, CLS)


def test_vocab_drops_rare_words_and_maps_them_to_unk():
    vocab = Vocab.build(["hello hello world"], min_count=2)
    assert "world" not in vocab.words
    assert vocab.id("world") == UNK
    assert len(vocab) == 4


def test_vocab_rejects_min_count_below_one():
    with pytest.raises(ValueError, match="min_count"):
        Vocab.build(["a"], min_count=0)


def test_encode_adds_cls_pads_and_truncates():
    vocab = Vocab.build(["a b c d e"], min_count=1)
    ids = encode(vocab, ["a b", "a b c d e zzz"], max_len=4)
    a, b, c = vocab.id("a"), vocab.id("b"), vocab.id("c")
    assert ids.dtype == torch.long
    assert ids.tolist() == [[CLS, a, b, PAD], [CLS, a, b, c]]


def test_encode_rejects_max_len_without_room_for_a_token():
    vocab = Vocab.build(["a"], min_count=1)
    with pytest.raises(ValueError, match="max_len"):
        encode(vocab, ["a"], max_len=1)


def test_token_groups_sort_tokens_by_kind_and_spam_rate():
    train = ([Message(True, "free prize now !")] * 5 + [Message(False, "see you at lunch 12")] * 5
             + [Message(False, "free lunch")])
    vocab = Vocab.build([m.text for m in train], min_count=1)
    groups = token_groups(vocab, train, min_count=5, spam_leaning=0.8, ham_leaning=0.2)
    name = {w: GROUPS[groups[vocab.id(w)]] for w in vocab.words if w != "[PAD]"}
    assert groups[PAD] == -1
    assert name["[CLS]"] == "[CLS]"
    assert name["[UNK]"] == "[UNK]"
    assert name["prize"] == name["now"] == "spam-leaning words"
    assert name["see"] == name["lunch"] == "ham-leaning words"
    assert name["free"] == "spam-leaning words"  # 5 of its 6 uses are in spam
    assert name["[NUM]"] == "numbers"
    assert name["!"] == "punctuation"


def test_token_groups_call_rare_words_other():
    train = [Message(True, "rare"), Message(False, "common")] * 2
    vocab = Vocab.build([m.text for m in train], min_count=1)
    groups = token_groups(vocab, train, min_count=5)
    assert GROUPS[groups[vocab.id("rare")]] == "other words"
