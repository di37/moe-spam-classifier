import pytest
import torch
import torch.nn.functional as F

from bert import BertConfig, TinyBert
from vocab import CLS, PAD


def small_model(n_experts=3, seed=0, max_len=24):
    torch.manual_seed(seed)
    config = BertConfig(vocab_size=50, max_len=max_len, d_model=16, n_heads=2, n_layers=2,
                        n_experts=n_experts, d_ff=32, dropout=0.1)
    return TinyBert(config).eval()


def message_ids(lengths, max_len, seed=1):
    """One row per length: [CLS], random word ids, then [PAD]s."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.full((len(lengths), max_len), PAD)
    for row, n in enumerate(lengths):
        ids[row, 0] = CLS
        ids[row, 1:n] = torch.randint(3, 50, (n - 1,), generator=gen)
    return ids


def test_output_shapes():
    model = small_model()
    out = model(message_ids([5, 9, 3], max_len=12))
    assert out.logits.shape == (3,)
    assert out.aux_loss.ndim == 0
    assert len(out.routing) == 2
    assert out.routing[0].probs.shape == (3, 12, 3)
    assert out.routing[1].mask.sum() == 5 + 9 + 3


def test_extra_padding_changes_nothing():
    model = small_model()
    short = message_ids([6, 4], max_len=8)
    long = torch.cat([short, torch.full((2, 12), PAD)], dim=1)
    with torch.no_grad():
        a, b = model(short), model(long)
    assert torch.allclose(a.logits, b.logits, atol=1e-6)
    assert torch.allclose(a.aux_loss, b.aux_loss, atol=1e-6)
    for ra, rb in zip(a.routing, b.routing, strict=True):
        assert torch.equal(ra.expert[ra.mask], rb.expert[rb.mask])
        assert torch.allclose(ra.probs[ra.mask], rb.probs[rb.mask], atol=1e-6)
        assert torch.allclose(ra.load(), rb.load())
    assert not torch.isnan(b.logits).any()


def test_a_message_scores_the_same_alone_or_in_a_batch():
    model = small_model()
    batch = message_ids([7, 3, 10, 5], max_len=12)
    with torch.no_grad():
        together = model(batch).logits
        alone = torch.stack([model(batch[i:i + 1]).logits[0] for i in range(4)])
    assert torch.allclose(together, alone, atol=1e-5)


def test_rows_must_start_with_cls():
    model = small_model()
    ids = message_ids([4], max_len=6)
    with pytest.raises(ValueError, match=r"\[CLS\]"):
        model(ids.roll(1, dims=1))


def test_rejects_sequences_longer_than_max_len():
    model = small_model(max_len=8)
    with pytest.raises(ValueError, match="max_len"):
        model(message_ids([4], max_len=9))


def test_every_router_learns_from_the_classification_loss_alone():
    model = small_model().train()
    out = model(message_ids([6, 9, 4, 8], max_len=10))
    F.binary_cross_entropy_with_logits(out.logits, torch.tensor([0.0, 1.0, 0.0, 1.0])).backward()
    for block in model.blocks:
        grad = block.moe.router.weight.grad
        assert grad is not None and grad.abs().sum() > 0


def test_one_expert_is_a_plain_dense_bert():
    model = small_model(n_experts=1)
    out = model(message_ids([5, 8], max_len=10))
    assert torch.allclose(out.aux_loss, torch.tensor(1.0))
    assert all(torch.equal(r.expert, torch.zeros_like(r.expert)) for r in out.routing)


def test_attention_weights_on_request_without_changing_the_result():
    model = small_model()
    ids = message_ids([6, 3], max_len=8)
    with torch.no_grad():
        plain, traced = model(ids), model(ids, need_weights=True)
    assert plain.attention is None
    assert torch.allclose(plain.logits, traced.logits, atol=1e-6)
    assert len(traced.attention) == 2
    weights = traced.attention[0]
    assert weights.shape == (2, 8, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5)
    assert torch.all(weights[1, :, 3:] == 0)  # nobody attends to padding


def test_same_seed_same_model():
    a, b = small_model(seed=4), small_model(seed=4)
    ids = message_ids([5, 7], max_len=8)
    with torch.no_grad():
        assert torch.equal(a(ids).logits, b(ids).logits)
