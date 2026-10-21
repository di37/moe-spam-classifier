import numpy as np
import pytest
import torch

from moe_spam.analysis.model_facts import ModelFacts
from moe_spam.analysis.routing import RouterProbe
from moe_spam.analysis.trace import MessageTracer
from moe_spam.data.vocab import Vocab
from moe_spam.model.bert import BertConfig, TinyBert

TEXTS = ["win a prize call 0800 now", "see you at lunch", "win cash now call 09061701461"]


def tiny_model(vocab, n_experts=3):
    torch.manual_seed(0)
    return TinyBert(BertConfig(vocab_size=len(vocab), max_len=32, d_model=16, n_heads=2,
                               n_experts=n_experts, d_ff=32)).eval()


def test_facts_read_the_models_sizes_and_count_its_parameters():
    vocab = Vocab.build(TEXTS, min_count=1)
    model = tiny_model(vocab)
    facts = ModelFacts.of(model)
    assert (facts.vocab_size, facts.max_len, facts.d_model, facts.n_heads) == (len(vocab), 32,
                                                                             16, 2)
    assert (facts.n_layers, facts.n_experts, facts.d_ff) == (2, 3, 32)
    assert facts.total_params == sum(p.numel() for p in model.parameters())
    assert facts.expert_params == 2 * 16 * 32 + 32 + 16  # two weight matrices and their biases
    assert facts.router_params == 16 * 3
    assert facts.moe_params == 3 * facts.expert_params + facts.router_params


def test_active_parameters_are_the_dense_model_plus_the_bigger_router():
    vocab = Vocab.build(TEXTS, min_count=1)
    moe, dense = ModelFacts.of(tiny_model(vocab)), ModelFacts.of(tiny_model(vocab, n_experts=1))
    # One expert runs per token, so a token touches what the dense model has, except that
    # each layer's router scores 3 experts instead of 1.
    extra_router = moe.n_layers * (moe.router_params - dense.router_params)
    assert moe.active_params == dense.total_params + extra_router
    assert dense.active_params == dense.total_params


def test_probs_are_every_tokens_router_probabilities():
    vocab = Vocab.build(TEXTS, min_count=1)
    model = tiny_model(vocab)
    probs = RouterProbe(model, vocab).probs(TEXTS[2], 0)
    trace = MessageTracer(model, vocab).trace(TEXTS[2])
    assert probs.shape == (len(trace.words), 3)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert tuple(probs.argmax(axis=1)) == trace.experts[0]
    assert np.allclose(probs[0], trace.cls_probs[0], atol=1e-6)
    with pytest.raises(ValueError, match="layer"):
        RouterProbe(model, vocab).probs(TEXTS[2], 5)
