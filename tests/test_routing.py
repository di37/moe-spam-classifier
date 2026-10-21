import numpy as np
import pytest
import torch
import torch.nn.functional as F

from moe_spam.analysis.routing import RouterProbe
from moe_spam.data.vocab import Vocab
from moe_spam.model.bert import BertConfig, TinyBert

TEXTS = ["win a prize call 0800 now", "see you at lunch", "win cash now call 09061701461"]
SPAM = TEXTS[2]


def tiny_setup(n_experts=3):
    vocab = Vocab.build(TEXTS, min_count=1)
    torch.manual_seed(0)
    model = TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                                n_experts=n_experts, d_ff=32)).eval()
    return model, vocab


def moe_outputs(model, vocab, text, layer):
    """The MoE layer's real output for ``text`` in ``layer``, caught by a hook."""
    caught = []
    hook = model.blocks[layer].moe.register_forward_hook(lambda m, a, out: caught.append(out[0]))
    with torch.no_grad():
        model(vocab.encode([text], model.config.max_len))
    hook.remove()
    return caught[0][0]


def loss_with_gate_scaled(model, vocab, text, is_spam, layer, position, factor):
    """The loss when one token's MoE output is multiplied by ``factor``."""
    def scale(module, args, out):
        y = out[0].clone()
        y[0, position] *= factor
        return (y, *out[1:])
    hook = model.blocks[layer].moe.register_forward_hook(scale)
    with torch.no_grad():
        logit = model(vocab.encode([text], model.config.max_len)).logits
    hook.remove()
    return F.binary_cross_entropy_with_logits(logit, torch.tensor([float(is_spam)])).item()


@pytest.mark.parametrize("layer, position", [(0, 0), (0, 3), (1, 5)])
def test_step_reproduces_the_router_and_the_layers_output(layer, position):
    model, vocab = tiny_setup()
    step = RouterProbe(model, vocab).step(SPAM, True, layer, position)
    assert step.word == ("[CLS]", "win", "cash", "now", "call", "09061701461")[position]
    assert step.weights.shape == (3, 16) and step.x.shape == (16,)
    assert np.allclose(step.logits, step.weights @ step.x, atol=1e-5)
    assert np.allclose(step.probs, np.exp(step.logits) / np.exp(step.logits).sum(), atol=1e-6)
    assert step.expert == int(np.argmax(step.probs))
    assert step.p_i == pytest.approx(step.probs.max())
    assert step.hidden.shape == (32,)
    assert np.allclose(step.y, step.p_i * step.ffn_out, atol=1e-6)
    real = moe_outputs(model, vocab, SPAM, layer)[position].numpy()
    assert np.allclose(step.y, real, atol=1e-5)


def test_loss_slope_is_the_derivative_of_the_loss_by_the_gate():
    model, vocab = tiny_setup()
    step = RouterProbe(model, vocab).step(SPAM, True, 0, 2)
    eps = 1e-3
    up = loss_with_gate_scaled(model, vocab, SPAM, True, 0, 2, (step.p_i + eps) / step.p_i)
    down = loss_with_gate_scaled(model, vocab, SPAM, True, 0, 2, (step.p_i - eps) / step.p_i)
    assert step.loss_slope == pytest.approx((up - down) / (2 * eps), rel=1e-2, abs=1e-6)


def test_step_refuses_positions_and_layers_that_do_not_exist():
    model, vocab = tiny_setup()
    probe = RouterProbe(model, vocab)
    with pytest.raises(ValueError, match="position"):
        probe.step(SPAM, True, 0, 6)  # 6 tokens: [CLS] and five words
    with pytest.raises(ValueError, match="layer"):
        probe.step(SPAM, True, 2, 0)


def test_clearest_is_the_word_the_router_is_surest_about():
    model, vocab = tiny_setup()
    probe = RouterProbe(model, vocab)
    position = probe.clearest(SPAM, 0)
    surest = [probe.step(SPAM, True, 0, p).p_i for p in range(1, 6)]
    assert position == 1 + int(np.argmax(surest))


def test_clearest_skips_punctuation():
    model, vocab = tiny_setup()
    probe = RouterProbe(model, vocab)
    text = "win , . ! 0800"
    words = ("[CLS]", "win", ",", ".", "!", "0800")
    assert words[probe.clearest(text, 0)] in ("win", "0800")  # numbers are words here
    with pytest.raises(ValueError, match="no words"):
        probe.clearest(", . !", 0)


def test_balance_matches_the_models_own_load_balancing_loss():
    model, vocab = tiny_setup()
    layers = RouterProbe(model, vocab).balance(TEXTS)
    with torch.no_grad():
        out = model(vocab.encode(TEXTS, model.config.max_len))
    assert len(layers) == 2
    for stats, routing in zip(layers, out.routing, strict=True):
        assert stats.load.sum() == pytest.approx(1)
        assert stats.mean_probs.sum() == pytest.approx(1)
        assert stats.aux == pytest.approx(routing.balance_loss().item())
        assert stats.aux == pytest.approx(3 * (stats.load * stats.mean_probs).sum())
    assert float(out.aux_loss) == pytest.approx(np.mean([s.aux for s in layers]))


def test_probe_gives_the_model_its_mode_back():
    model, vocab = tiny_setup()
    model.train()
    probe = RouterProbe(model, vocab)
    probe.step(SPAM, True, 0, 1)
    probe.balance(TEXTS)
    assert model.training
    assert all(p.grad is None for p in model.parameters())  # the slope leaves no gradients
