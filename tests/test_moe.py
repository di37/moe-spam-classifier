import pytest
import torch
import torch.nn.functional as F
from torch import nn

from moe_spam.model.moe import MoELayer, Routing

D_MODEL, D_FF = 8, 16


def make_layer(n_experts=3, router_std=0.02, seed=0):
    torch.manual_seed(seed)
    return MoELayer(D_MODEL, n_experts, D_FF, router_std=router_std)


def make_batch(lengths=(6, 4), max_len=6, seed=1):
    """Random token vectors, and a mask with lengths[b] real tokens at the start of row b."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(lengths), max_len, D_MODEL, generator=g)
    mask = torch.arange(max_len) < torch.tensor(lengths)[:, None]
    return x, mask


def dense_reference(layer, x, mask):
    """Run every expert on every token, then keep gate * output of each token's top-1 expert."""
    probs = F.softmax(layer.router(x), dim=-1)
    gate, top1 = probs.max(dim=-1)
    pick = F.one_hot(top1, layer.n_experts).to(x.dtype)            # [B, L, E]
    outs = torch.stack([ffn(x) for ffn in layer.experts], dim=-2)  # [B, L, E, d_model]
    dense = (pick[..., None] * gate[..., None, None] * outs).sum(dim=-2)
    return dense * mask[..., None]


def starve_last_expert(layer):
    """
    Router weights v, -v and 0 for three experts: the logits are a, -a and 0, and
    max(a, -a) > 0 unless a is exactly 0, so the last expert never wins.
    """
    v = torch.randn(D_MODEL)
    with torch.no_grad():
        layer.router.weight.copy_(torch.stack([v, -v, torch.zeros(D_MODEL)]))


def uniform_routing(n_experts, n_tokens):
    """Uniform router probabilities, with the top-1 cycling through the experts in turn."""
    probs = torch.full((1, n_tokens, n_experts), 1 / n_experts)
    expert = torch.arange(n_tokens).remainder(n_experts)[None, :]
    return probs, expert


# --- construction ---------------------------------------------------------------

@pytest.mark.parametrize("router_std", [0.02, 0.5])
def test_router_is_a_bias_free_linear_with_small_random_weights(router_std):
    torch.manual_seed(0)
    layer = MoELayer(64, 3, 256, router_std=router_std)
    w = layer.router.weight
    assert layer.router.bias is None
    assert w.shape == (3, 64)
    assert w.mean().item() == pytest.approx(0, abs=router_std / 4)
    assert w.std().item() == pytest.approx(router_std, rel=0.2)


def test_each_expert_is_a_two_layer_gelu_ffn():
    layer = make_layer(n_experts=4)
    assert len(layer.experts) == 4
    for ffn in layer.experts:
        first, act, second = ffn
        assert isinstance(act, nn.GELU)
        assert (first.in_features, first.out_features) == (D_MODEL, D_FF)
        assert (second.in_features, second.out_features) == (D_FF, D_MODEL)


def test_n_experts_must_be_at_least_one():
    with pytest.raises(ValueError, match="n_experts"):
        MoELayer(D_MODEL, 0, D_FF)


# --- forward --------------------------------------------------------------------

def test_forward_returns_output_aux_loss_and_routing():
    x, mask = make_batch()
    out, aux, routing = make_layer()(x, mask)
    assert out.shape == x.shape
    assert aux.ndim == 0
    assert routing.probs.shape == (2, 6, 3)
    assert torch.allclose(routing.probs.sum(dim=-1), torch.ones(2, 6))
    assert routing.expert.shape == (2, 6)
    assert routing.expert.dtype == torch.long
    assert torch.equal(routing.mask, mask)


def test_forward_does_not_change_its_inputs():
    x, mask = make_batch()
    x_before, mask_before = x.clone(), mask.clone()
    make_layer()(x, mask)
    assert torch.equal(x, x_before)
    assert torch.equal(mask, mask_before)


def test_forward_rejects_a_mask_of_the_wrong_shape_or_dtype():
    layer = make_layer()
    x, mask = make_batch()
    with pytest.raises(ValueError, match="mask"):
        layer(x, mask[:, :-1])
    with pytest.raises(ValueError, match="mask"):
        layer(x, mask.long())


def test_one_expert_is_exactly_the_dense_ffn_on_real_tokens():
    layer = make_layer(n_experts=1)
    x, mask = make_batch()
    out, _, routing = layer(x, mask)
    assert torch.all(routing.probs == 1)  # a softmax over one expert: the gate is exactly 1
    assert torch.allclose(out[mask], layer.experts[0](x)[mask], atol=1e-6)
    assert torch.all(out[~mask] == 0)


def test_sparse_dispatch_matches_a_dense_reference():
    layer = make_layer(router_std=1.0)
    x, mask = make_batch(lengths=(6, 4, 5))
    out, _, routing = layer(x, mask)
    assert torch.all(routing.load() > 0)  # every expert gets tokens in this batch
    assert torch.allclose(out, dense_reference(layer, x, mask), atol=1e-6)


def test_sparse_dispatch_matches_a_dense_reference_when_an_expert_gets_no_tokens():
    layer = make_layer()
    starve_last_expert(layer)
    x, mask = make_batch()
    out, _, routing = layer(x, mask)
    assert routing.load()[2] == 0
    assert torch.all(routing.load()[:2] > 0)
    assert torch.allclose(out, dense_reference(layer, x, mask), atol=1e-6)


def test_each_expert_runs_only_on_the_real_tokens_it_won():
    layer = make_layer(router_std=1.0)
    x, mask = make_batch(lengths=(6, 4, 5))
    rows_seen = []
    for ffn in layer.experts:
        ffn.register_forward_hook(lambda _ffn, inputs, _out: rows_seen.append(len(inputs[0])))
    _, _, routing = layer(x, mask)
    assert rows_seen == torch.bincount(routing.expert[mask], minlength=3).tolist()
    assert sum(rows_seen) == mask.sum()  # no pad ever reaches an expert


def test_forward_works_under_autocast():
    layer = make_layer()
    x, mask = make_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, aux, _ = layer(x, mask)
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux)
    assert torch.allclose(out, layer(x, mask)[0], atol=0.05)


def test_an_expert_that_gets_no_tokens_gets_no_gradient():
    layer = make_layer()
    starve_last_expert(layer)
    x, mask = make_batch()
    out, _, _ = layer(x, mask)
    out.pow(2).sum().backward()
    for p in layer.experts[2].parameters():
        assert p.grad is None or torch.all(p.grad == 0)
    assert layer.experts[0][0].weight.grad.abs().sum() > 0


# --- padding --------------------------------------------------------------------

@pytest.mark.parametrize("pad_value", [1e4, float("nan")])
def test_pad_values_change_nothing_and_pads_give_zero_output(pad_value):
    layer = make_layer()
    x, mask = make_batch(lengths=(6, 4, 0))  # the last row is all padding
    other = torch.where(mask[..., None], x, torch.full_like(x, pad_value))
    out, aux, routing = layer(x, mask)
    other_out, other_aux, other_routing = layer(other, mask)
    assert torch.equal(other_out[mask], out[mask])
    assert torch.equal(other_aux, aux)
    assert torch.equal(other_routing.load(), routing.load())
    assert torch.all(other_out[~mask] == 0)
    assert not torch.isnan(other_out).any()
    assert not torch.isnan(other_aux)


def test_nan_pads_give_finite_gradients_and_none_flows_back_into_the_pads():
    layer = make_layer()
    x, mask = make_batch()
    x = torch.where(mask[..., None], x, torch.full_like(x, float("nan"))).requires_grad_()
    out, aux, _ = layer(x, mask)
    (out.pow(2).sum() + aux).backward()
    assert torch.isfinite(layer.router.weight.grad).all()
    assert torch.isfinite(x.grad).all()
    assert torch.all(x.grad[~mask] == 0)


# --- load-balancing loss ----------------------------------------------------------

@pytest.mark.parametrize("n_experts", [1, 2, 3, 4])
def test_load_balancing_loss_is_one_when_routing_is_perfectly_balanced(n_experts):
    probs, expert = uniform_routing(n_experts, n_tokens=2 * n_experts)
    mask = torch.ones_like(expert, dtype=torch.bool)
    loss = Routing(probs, expert, mask).balance_loss()
    assert loss.item() == pytest.approx(1.0)


@pytest.mark.parametrize("n_experts", [1, 2, 3, 4])
def test_load_balancing_loss_is_n_experts_when_routing_collapses_onto_one_expert(n_experts):
    expert = torch.zeros(1, 5, dtype=torch.long)
    probs = F.one_hot(expert, n_experts).float()
    mask = torch.ones_like(expert, dtype=torch.bool)
    loss = Routing(probs, expert, mask).balance_loss()
    assert loss.item() == pytest.approx(n_experts)


HAND_PROBS = torch.tensor([[[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]]])
HAND_EXPERT = torch.tensor([[0, 1]])


def test_load_balancing_loss_matches_a_hand_computed_value():
    # f = [1/2, 1/2, 0] and P = [0.4, 0.4, 0.2], so the loss is 3 * (0.2 + 0.2 + 0) = 1.2
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = Routing(HAND_PROBS, HAND_EXPERT, mask).balance_loss()
    assert loss.item() == pytest.approx(1.2)


def test_load_balancing_loss_ignores_pads():
    # Two pads sure of expert 2: counting them in f would give 0.9, counting them in P 0.6
    pad_expert = torch.full((1, 2), 2)
    probs = torch.cat([HAND_PROBS, F.one_hot(pad_expert, 3).float()], dim=1)
    expert = torch.cat([HAND_EXPERT, pad_expert], dim=1)
    mask = torch.tensor([[True, True, False, False]])
    loss = Routing(probs, expert, mask).balance_loss()
    assert loss.item() == pytest.approx(1.2)


def test_load_balancing_loss_needs_real_tokens():
    probs, expert = uniform_routing(3, n_tokens=4)
    with pytest.raises(ValueError, match="real tokens"):
        Routing(probs, expert, torch.zeros_like(expert, dtype=torch.bool)).balance_loss()


# --- gradients ------------------------------------------------------------------

def test_router_learns_from_the_aux_loss_alone():
    layer = make_layer()
    x, mask = make_batch()  # 10 real tokens cannot split evenly over 3 experts
    _, aux, _ = layer(x, mask)
    aux.backward()
    assert layer.router.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in layer.experts.parameters())


def test_router_learns_from_a_task_loss_on_the_output_through_the_gate():
    layer = make_layer()
    x, mask = make_batch()
    out, _, _ = layer(x, mask)
    out.pow(2).sum().backward()
    assert layer.router.weight.grad.abs().sum() > 0


# --- Routing.load ---------------------------------------------------------------

def test_load_is_the_share_of_real_tokens_sent_to_each_expert():
    expert = torch.tensor([[0, 0, 1, 2]])
    probs = torch.full((1, 4, 3), 1 / 3)
    mask = torch.ones(1, 4, dtype=torch.bool)
    load = Routing(probs=probs, expert=expert, mask=mask).load()
    assert torch.allclose(load, torch.tensor([0.5, 0.25, 0.25]))


def test_load_ignores_pads():
    expert = torch.tensor([[0, 1, 2, 2, 2, 2]])
    probs = torch.full((1, 6, 3), 1 / 3)
    mask = torch.tensor([[True, True, True, False, False, False]])
    load = Routing(probs=probs, expert=expert, mask=mask).load()
    assert torch.allclose(load, torch.full((3,), 1 / 3))


def test_load_after_a_forward_pass_covers_every_expert_and_sums_to_one():
    x, mask = make_batch()
    _, _, routing = make_layer()(x, mask)
    load = routing.load()
    assert load.shape == (3,)
    assert load.sum().item() == pytest.approx(1.0)
