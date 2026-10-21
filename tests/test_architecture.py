import os

import numpy as np
import pytest
import torch
from PIL import Image

from architecture_panels import token_layout
from architecture_video import beats_for, caption, render_walkthrough, step_titles
from bert import BertConfig, TinyBert
from frames import FIGURE_PX
from sms_data import Message
from vocab import Vocab, encode
from walkthrough import MAX_TOKENS, Pick, pick_messages, preference, trace_message

TEXTS = ["win a prize call 0800 now", "see you at lunch", "win cash now call 09061701461"]


PANEL = (104.45, 20.25)  # the detail panel of a real frame, in characters and lines


def tiny_model(vocab, n_experts=3, n_layers=2):
    torch.manual_seed(0)
    return TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                               n_layers=n_layers, n_experts=n_experts, d_ff=32)).eval()


def test_trace_follows_one_message_through_the_model():
    vocab = Vocab.build(TEXTS, min_count=1)
    model = tiny_model(vocab)
    text = "WIN a prize, call 0800 now!"
    trace = trace_message(model, vocab, text)
    assert trace.words == ("[CLS]", "win", "a", "prize", ",", "call", "0800", "now", "!")
    assert trace.tokens == ("[CLS]", "win", "a", "prize", "[UNK]", "call", "[NUM]", "now",
                            "[UNK]")
    n = len(trace.words)
    assert trace.embeddings.shape == (n, 16)
    assert len(trace.cls_attention) == len(trace.experts) == len(trace.cls_probs) == 2
    for weights, experts, probs in zip(trace.cls_attention, trace.experts, trace.cls_probs,
                                       strict=True):
        assert weights.shape == (n,) and weights.sum() == pytest.approx(1, abs=1e-5)
        assert len(experts) == n and all(0 <= e < 3 for e in experts)
        assert probs.sum() == pytest.approx(1, abs=1e-5)
        assert experts[0] == probs.argmax()
    with torch.no_grad():
        logit = model(encode(vocab, [text], 32)).logits[0]
    assert trace.p_spam == pytest.approx(torch.sigmoid(logit).item(), abs=1e-6)


def test_pick_messages_returns_a_spam_and_a_ham_with_known_words():
    messages = [Message(True, "win cash now call 09061701461"),
                Message(False, "see you at lunch"),
                Message(True, "win a prize zzz"),  # has an unknown word
                Message(False, "see you")]
    vocab = Vocab.build(TEXTS, min_count=1)
    picks = pick_messages(tiny_model(vocab), vocab, messages)
    assert picks["spam"].text == "win cash now call 09061701461"
    assert picks["ham"].text in ("see you at lunch", "see you")
    assert 1 <= picks["spam"].same <= picks["spam"].total == 2
    assert picks["spam"].trace.experts[-1][0] in range(3)


def test_pick_messages_never_picks_a_message_too_long_to_show():
    long_spam = " ".join(["win cash now"] * 10)  # 30 tokens
    vocab = Vocab.build(TEXTS, min_count=1)
    messages = [Message(True, long_spam), Message(False, "see you at lunch")]
    with pytest.raises(ValueError, match="spam"):
        pick_messages(tiny_model(vocab), vocab, messages)
    assert len(long_spam.split()) > MAX_TOKENS


def test_pick_messages_needs_both_labels():
    vocab = Vocab.build(TEXTS, min_count=1)
    with pytest.raises(ValueError, match="ham"):
        pick_messages(tiny_model(vocab), vocab, [Message(True, TEXTS[0])])
    with pytest.raises(ValueError, match="no messages"):
        pick_messages(tiny_model(vocab), vocab, [])


def test_preference_puts_right_typical_and_plain_messages_first():
    plain = preference(wrong=False, unusual=False, n_tokens=12, n_punctuation=0, n_chars=60)
    assert plain < preference(wrong=True, unusual=False, n_tokens=12, n_punctuation=0,
                              n_chars=60)
    assert plain < preference(wrong=False, unusual=True, n_tokens=12, n_punctuation=0,
                              n_chars=60)
    assert plain < preference(wrong=False, unusual=False, n_tokens=4, n_punctuation=0,
                              n_chars=20)  # too short to show much
    assert plain < preference(wrong=False, unusual=False, n_tokens=12, n_punctuation=3,
                              n_chars=50)
    assert plain < preference(wrong=False, unusual=False, n_tokens=12, n_punctuation=0,
                              n_chars=70)


def test_token_layout_keeps_tokens_inside_the_panel_and_their_expert_boxes():
    words = ("[CLS]", "congratulations", "you've", "won", ".", "call", "09066660100", "now")
    experts = ((0, 1, 2, 1, 1, 0, 2, 2), (1, 1, 1, 0, 2, 2, 0, 1))
    layout = token_layout("Congratulations you've won. Call 09066660100 now", words, experts,
                          width=60, height=20)
    assert all(x + len(w) <= 60 for (x, _), w in zip(layout.home, words, strict=True))
    assert len({y for _, y in layout.home}) == 2  # wrapped onto a second row
    for layer, placed in enumerate(layout.bins):
        for (x, y), word, expert in zip(placed, words, experts[layer], strict=True):
            x0, y0, x1, y1 = layout.bin_boxes[expert]
            assert x0 < x and x + len(word) <= x1 and y0 < y < y1


def test_token_layout_refuses_a_message_that_does_not_fit():
    words = ("[CLS]", *(["unbelievable", "opportunity"] * 20))
    experts = (tuple(i % 3 for i in range(len(words))),) * 2
    with pytest.raises(ValueError, match="does not fit"):
        token_layout(" ".join(words[1:]), words, experts, *PANEL)


def test_a_long_message_still_fits_the_real_panel():
    text = ("We tried to call you re your reply to our sms for a video mobile 750 mins "
            "UNLIMITED TEXT free camcorder")
    words = ("[CLS]", *text.lower().split())
    assert len(words) - 1 <= MAX_TOKENS
    for expert in range(3):  # even if one expert takes every token
        token_layout(text, words, ((expert,) * len(words),) * 2, *PANEL)


def test_step_numbers_follow_the_steps_shown():
    assert step_titles("spam")["embed"] == "3. Embeddings"
    ham = step_titles("ham")
    assert "embed" not in ham
    assert ham["tokenize"] == "2. Tokenizer"
    assert ham["attn1"] == "3. Layer 1: self-attention"
    assert ham["route1"] == ham["back1"]


def test_route_caption_gives_exact_counts():
    vocab = Vocab.build(TEXTS, min_count=1)
    trace = trace_message(tiny_model(vocab), vocab, TEXTS[1])
    assert "899 of the 903 ham" in caption("route2", Pick(TEXTS[1], trace, 899, 903), "ham")
    assert "all 903 ham" in caption("route2", Pick(TEXTS[1], trace, 903, 903), "ham")


def test_render_writes_every_frame_and_holds(tmp_path):
    vocab = Vocab.build(TEXTS, min_count=1)
    model = tiny_model(vocab)
    picks = pick_messages(model, vocab, [Message(True, TEXTS[2]), Message(False, TEXTS[1])])
    folder = tmp_path / "frames"
    n = render_walkthrough(picks, str(folder), fps=2)
    files = sorted(os.listdir(folder))
    assert len(files) == n == sum(f + h for label in ("spam", "ham")
                                  for _, f, h in beats_for(label, fps=2))
    with Image.open(folder / files[-1]) as img:
        assert img.size == FIGURE_PX
    assert np.array_equal(np.asarray(Image.open(folder / files[-1])),
                          np.asarray(Image.open(folder / files[-2])))  # a held frame


@pytest.mark.parametrize("sizes", [{"n_experts": 2}, {"n_layers": 1}, {"n_layers": 3}])
def test_render_needs_two_layers_of_three_experts(tmp_path, sizes):
    vocab = Vocab.build(TEXTS, min_count=1)
    model = tiny_model(vocab, **sizes)
    picks = pick_messages(model, vocab, [Message(True, TEXTS[2]), Message(False, TEXTS[1])])
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "0000000.jpg").write_bytes(b"old frame")
    with pytest.raises(ValueError, match="2 layers of 3 experts"):
        render_walkthrough(picks, str(tmp_path / "frames"), fps=2)
    assert (tmp_path / "frames" / "0000000.jpg").exists()  # checked before clearing
