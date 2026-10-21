import os

import numpy as np
import pytest
import torch
from PIL import Image

from moe_spam.analysis.examples import MAX_TOKENS, ExampleFinder, Pick, preference
from moe_spam.analysis.trace import MessageTracer
from moe_spam.data.sms import Message
from moe_spam.data.vocab import Vocab
from moe_spam.errors import MoeSpamError
from moe_spam.model.bert import BertConfig, TinyBert
from moe_spam.viz.architecture.layout import TokenLayout, fits_on_screen
from moe_spam.viz.architecture.script import BEATS, caption, step_titles
from moe_spam.viz.architecture.video import ArchitectureVideo
from moe_spam.viz.style import FIGURE_PX

TEXTS = ["win a prize call 0800 now", "see you at lunch", "win cash now call 09061701461"]
PANEL = (104.45, 20.25)  # the detail panel of a real frame, in characters and lines


def tiny_model(vocab, n_experts=3, n_layers=2):
    torch.manual_seed(0)
    return TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                               n_layers=n_layers, n_experts=n_experts, d_ff=32)).eval()


def picks_of(model, vocab):
    return ExampleFinder(vocab).ranked(model, [Message(True, TEXTS[2]), Message(False, TEXTS[1])])


def test_trace_follows_one_message_through_the_model():
    vocab = Vocab.build(TEXTS, min_count=1)
    model = tiny_model(vocab)
    text = "WIN a prize, call 0800 now!"
    trace = MessageTracer(model, vocab).trace(text)
    assert trace.words == ("[CLS]", "win", "a", "prize", ",", "call", "0800", "now", "!")
    assert trace.tokens == ("[CLS]", "win", "a", "prize", "[UNK]", "call", "[NUM]", "now",
                            "[UNK]")
    assert (trace.d_model, trace.n_experts) == (16, 3)
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
        logit = model(vocab.encode([text], 32)).logits[0]
    assert trace.p_spam == pytest.approx(torch.sigmoid(logit).item(), abs=1e-6)


def test_ranked_returns_a_spam_and_a_ham_with_known_words():
    messages = [Message(True, "win cash now call 09061701461"),
                Message(False, "see you at lunch"),
                Message(True, "win a prize zzz"),  # has an unknown word
                Message(False, "see you")]
    vocab = Vocab.build(TEXTS, min_count=1)
    picks = ExampleFinder(vocab).ranked(tiny_model(vocab), messages)
    assert picks["spam"].text == "win cash now call 09061701461"
    assert picks["ham"].text in ("see you at lunch", "see you")
    assert 1 <= picks["spam"].same <= picks["spam"].total == 2
    assert picks["spam"].trace.experts[-1][0] in range(3)


def test_ranked_skips_messages_the_check_turns_down():
    messages = [Message(True, TEXTS[2]), Message(True, TEXTS[0]), Message(False, TEXTS[1])]
    vocab = Vocab.build(TEXTS, min_count=1)
    picks = ExampleFinder(vocab).ranked(tiny_model(vocab), messages,
                                        fits=lambda trace: trace.text != TEXTS[2])
    assert picks["spam"].text == TEXTS[0]


def test_ranked_never_picks_a_message_too_long_to_show():
    long_spam = " ".join(["win cash now"] * 10)  # 30 tokens
    vocab = Vocab.build(TEXTS, min_count=1)
    messages = [Message(True, long_spam), Message(False, "see you at lunch")]
    with pytest.raises(MoeSpamError, match="spam"):
        ExampleFinder(vocab).ranked(tiny_model(vocab), messages)
    assert len(long_spam.split()) > MAX_TOKENS


def test_ranked_needs_both_labels():
    vocab = Vocab.build(TEXTS, min_count=1)
    finder = ExampleFinder(vocab)
    with pytest.raises(MoeSpamError, match="ham"):
        finder.ranked(tiny_model(vocab), [Message(True, TEXTS[0])])
    with pytest.raises(MoeSpamError, match="no messages"):
        finder.ranked(tiny_model(vocab), [])


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
    layout = TokenLayout.build("Congratulations you've won. Call 09066660100 now", words,
                               experts, width=60, height=20)
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
        TokenLayout.build(" ".join(words[1:]), words, experts, *PANEL)


def test_a_long_message_still_fits_the_real_panel():
    text = ("We tried to call you re your reply to our sms for a video mobile 750 mins "
            "UNLIMITED TEXT free camcorder")
    words = ("[CLS]", *text.lower().split())
    assert len(words) - 1 <= MAX_TOKENS
    for expert in range(3):  # even if one expert takes every token
        TokenLayout.build(text, words, ((expert,) * len(words),) * 2, *PANEL)


def test_fits_on_screen_answers_for_a_trace():
    vocab = Vocab.build(TEXTS, min_count=1)
    tracer = MessageTracer(tiny_model(vocab), vocab)
    assert fits_on_screen(tracer.trace(TEXTS[1]), PANEL)
    assert not fits_on_screen(tracer.trace(" ".join(["win cash now"] * 10)), (30, 10))


def test_step_numbers_follow_the_steps_shown():
    assert step_titles("spam")["embed"] == "3. Embeddings"
    ham = step_titles("ham")
    assert "embed" not in ham
    assert ham["tokenize"] == "2. Tokenizer"
    assert ham["attn1"] == "3. Layer 1: self-attention"
    assert ham["route1"] == ham["back1"]


def test_route_caption_gives_exact_counts():
    vocab = Vocab.build(TEXTS, min_count=1)
    trace = MessageTracer(tiny_model(vocab), vocab).trace(TEXTS[1])
    assert "899 of the 903 ham" in caption("route2", Pick(TEXTS[1], trace, 899, 903), "ham")
    assert "all 903 ham" in caption("route2", Pick(TEXTS[1], trace, 903, 903), "ham")
    assert "16 numbers" in caption("embed", Pick(TEXTS[1], trace, 1, 1), "spam")


def test_render_writes_every_frame_and_holds(tmp_path):
    vocab = Vocab.build(TEXTS, min_count=1)
    folder = tmp_path / "frames"
    n = ArchitectureVideo(picks_of(tiny_model(vocab), vocab), fps=2).render(folder)
    files = sorted(os.listdir(folder))
    assert len(files) == n == sum(sum(beat.frames(fps=2)) for label in ("spam", "ham")
                                  for beat in BEATS[label])
    with Image.open(folder / files[-1]) as img:
        assert img.size == FIGURE_PX
    assert np.array_equal(np.asarray(Image.open(folder / files[-1])),
                          np.asarray(Image.open(folder / files[-2])))  # a held frame


@pytest.mark.parametrize("sizes", [{"n_experts": 2}, {"n_layers": 1}, {"n_layers": 3}])
def test_render_needs_two_layers_of_three_experts(tmp_path, sizes):
    vocab = Vocab.build(TEXTS, min_count=1)
    video = ArchitectureVideo(picks_of(tiny_model(vocab, **sizes), vocab), fps=2)
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "0000000.jpg").write_bytes(b"old frame")
    with pytest.raises(ValueError, match="2 layers of 3 experts"):
        video.render(tmp_path / "frames")
    assert (tmp_path / "frames" / "0000000.jpg").exists()  # checked before clearing
