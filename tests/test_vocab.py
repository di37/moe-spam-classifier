import numpy as np
import pytest
import torch

from moe_spam.data.sms import Message
from moe_spam.data.tokenizer import CLS, PAD, TOKENIZER, UNK, Tokenizer
from moe_spam.data.vocab import GROUPS, TokenGroups, Vocab


def test_tokenize_splits_words_numbers_and_punctuation():
    assert TOKENIZER.tokenize("WINNER!! Call 09061701461 now, 4 a £900 prize") == [
        "winner", "!", "!", "call", "[NUM5+]", "now", ",", "[NUM]", "a", "£", "[NUM]", "prize"]


def test_tokenize_keeps_apostrophes_inside_words():
    assert TOKENIZER.tokenize("I'd say it's ok") == ["i'd", "say", "it's", "ok"]


def test_pieces_are_the_lowercased_surface_forms_of_the_tokens():
    text = "Txt 87121 NOW!"
    assert TOKENIZER.pieces(text) == ["txt", "87121", "now", "!"]
    assert len(TOKENIZER.pieces(text)) == len(TOKENIZER.tokenize(text))


def test_long_number_sets_where_short_codes_start():
    assert Tokenizer(long_number=3).tokenize("12 123") == ["[NUM]", "[NUM5+]"]


def test_one_punctuation_rule():
    assert Tokenizer.is_punctuation("!") and Tokenizer.is_punctuation("[")
    assert Tokenizer.is_punctuation("£")
    assert not Tokenizer.is_punctuation("word")
    assert not Tokenizer.is_punctuation("[NUM]") and not Tokenizer.is_punctuation("[CLS]")
    assert Tokenizer.is_number("[NUM5+]") and not Tokenizer.is_number("5")


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
    assert vocab.knows(["hello"]) and not vocab.knows(["hello", "world"])


def test_vocab_rejects_min_count_below_one():
    with pytest.raises(ValueError, match="min_count"):
        Vocab.build(["a"], min_count=0)


def test_encode_adds_cls_pads_and_truncates():
    vocab = Vocab.build(["a b c d e"], min_count=1)
    ids = vocab.encode(["a b", "a b c d e zzz"], max_len=4)
    a, b, c = vocab.id("a"), vocab.id("b"), vocab.id("c")
    assert ids.dtype == torch.long
    assert ids.tolist() == [[CLS, a, b, PAD], [CLS, a, b, c]]


def test_encode_rejects_max_len_without_room_for_a_token():
    vocab = Vocab.build(["a"], min_count=1)
    with pytest.raises(ValueError, match="max_len"):
        vocab.encode(["a"], max_len=1)


def test_token_groups_sort_tokens_by_kind_and_spam_rate():
    train = ([Message(True, "free prize now !")] * 5 + [Message(False, "see you at lunch 12")] * 5
             + [Message(False, "free lunch")])
    vocab = Vocab.build([m.text for m in train], min_count=1)
    groups = TokenGroups.from_messages(vocab, train, min_count=5, spam_leaning=0.8,
                                       ham_leaning=0.2)
    name = {w: GROUPS[groups.ids[vocab.id(w)]] for w in vocab.words if w != "[PAD]"}
    assert groups.ids[PAD] == -1
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
    groups = TokenGroups.from_messages(vocab, train, min_count=5)
    assert GROUPS[groups.ids[vocab.id("rare")]] == "other words"


def test_token_group_labels_count_the_real_tokens():
    train = [Message(True, "win 12"), Message(False, "hi")]
    vocab = Vocab.build([m.text for m in train], min_count=1)
    groups = TokenGroups.from_messages(vocab, train)
    labels = groups.labels(vocab.encode(["win 12", "hi"], max_len=5))
    assert labels[GROUPS.index("[CLS]")] == "[CLS] (2)"
    assert labels[GROUPS.index("numbers")] == "numbers (1)"


def test_expert_shares_rows_sum_to_one_and_empty_groups_are_nan():
    groups = torch.tensor([0, 0, 0, 2, 2])
    experts = torch.tensor([0, 1, 1, 2, 2])
    shares = TokenGroups.expert_shares(groups, experts, n_experts=3)
    assert shares.shape == (len(GROUPS), 3)
    assert np.allclose(shares[0], [1 / 3, 2 / 3, 0])
    assert np.isnan(shares[1]).all()
    assert np.allclose(shares[2], [0, 0, 1])
