"""Synthetic numerical tests; do not assert that a T′ improves recognition."""
import pytest
import torch
from torch.nn import functional as F

from experiments.hospital_temporal.model import (
    TemporalVariant, VARIANTS, PROTECTED, temporal_contribution,
)


def batch(lengths=(5, 1), dims=8):
    valid = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
    a = torch.randn(len(lengths), max(lengths), 7)
    x = torch.cat((torch.randn(len(lengths), max(lengths), dims), a), -1)
    return x, a, valid


@pytest.mark.parametrize("variant", VARIANTS)
def test_initial_A_and_finite_updates(variant):
    torch.manual_seed(2)
    model = TemporalVariant(variant, 4, 4, 8, dropout=0)
    x, a, mask = batch()
    x.requires_grad_(); a.requires_grad_()
    model.eval()
    assert torch.equal(model(x, a, mask), a)
    before = model.base.classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = model(x, a, mask)
        labels = torch.tensor([0, 1, 2, 3, 4, 5])
        loss = F.cross_entropy(value[mask], labels)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.base.classifier.weight.grad.abs().sum() > 0
        if step:
            assert model.base.stem[0].weight.grad.abs().sum() > 0
            assert sum(p.grad.abs().sum() for p in model.base.blocks.parameters() if p.grad is not None) > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert not torch.equal(before, model.base.classifier.weight)
    assert x.grad is None and a.grad is None


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_kernel_center_bias_padding_dilation_equivalence(dilation):
    model = TemporalVariant("relative_tcn", 4, 4, 8, dropout=0)
    conv = model.base.blocks[0].conv
    x = torch.randn(2, 8, 13)
    valid = torch.arange(13)[None] < torch.tensor([13, 7])[:, None]
    x *= valid[:, None]
    expected = F.conv1d(x, conv.weight, conv.bias, padding=dilation, dilation=dilation)
    actual = temporal_contribution(x, valid, conv, dilation)
    torch.testing.assert_close(actual.transpose(1, 2)[valid], expected.transpose(1, 2)[valid], atol=2e-6, rtol=2e-6)
    center = temporal_contribution(x, valid, conv, dilation, center_only=True)
    torch.testing.assert_close(center, F.conv1d(x, conv.weight[:, :, 1:2], conv.bias))


def test_relative_constant_neighbors_and_missing_padding():
    model = TemporalVariant("relative_tcn", 4, 4, 8, dropout=0)
    conv = model.base.blocks[0].conv
    x = torch.ones(1, 8, 5)
    valid = torch.ones(1, 5, dtype=torch.bool)
    center = F.conv1d(x, conv.weight[:, :, 1:2], conv.bias)
    torch.testing.assert_close(temporal_contribution(x, valid, conv, 2, relative=True), center)
    x[:, :, 2] = 2
    changed = temporal_contribution(x, valid, conv, 2, relative=True)
    assert not torch.equal(changed, F.conv1d(x, conv.weight[:, :, 1:2], conv.bias))


def test_plain_model_is_exact_legacy_forward_not_substituted_reference():
    model = TemporalVariant("plain_tcn", 4, 4, 8, dropout=0).eval()
    torch.nn.init.normal_(model.base.classifier.weight)
    x, a, valid = batch()
    assert torch.equal(model(x, a, valid), model.base(x, a, valid))
    assert not torch.equal(model(x, a, valid)[1, 0], a[1, 0])


@pytest.mark.parametrize("variant", sorted(PROTECTED))
def test_singletons_remain_exact_A_after_nonzero_classifier(variant):
    model = TemporalVariant(variant, 4, 4, 8, dropout=0).eval()
    torch.nn.init.normal_(model.base.classifier.weight)
    x, a, valid = batch()
    assert torch.equal(model(x, a, valid)[1, 0], a[1, 0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_padding_does_not_change_short_segment(variant):
    model = TemporalVariant(variant, 4, 4, 8, dropout=0).eval()
    torch.nn.init.normal_(model.base.classifier.weight)
    x, a, valid = batch((11, 3))
    alone = model(x[1:2, :3], a[1:2, :3], valid[1:2, :3])
    padded = model(x, a, valid)[1:2, :3]
    torch.testing.assert_close(alone, padded, atol=3e-6, rtol=3e-6)


def test_shared_parameters_random_state_and_zero_mode():
    torch.manual_seed(7)
    before = torch.random.get_rng_state().clone()
    models = [TemporalVariant(v) for v in VARIANTS if v != "clip_only"]
    assert torch.equal(before, torch.random.get_rng_state())
    assert len({m.parameter_counts()["trainable"] for m in models}) == 1
    for model in models[1:]:
        assert all(torch.equal(v, model.base.state_dict()[k]) for k, v in models[0].base.state_dict().items())
    x, a, valid = batch()
    small = TemporalVariant("relative_tcn", 4, 4, 8)
    assert torch.equal(small(x, a, valid, mode="zero_correction"), a)


def test_paired_A_and_bad_masks_rejected():
    model = TemporalVariant("multispan_tcn", 4, 4, 8)
    x, a, valid = batch()
    with pytest.raises(ValueError): model(x, a+1, valid)
    valid[0, 1] = False
    with pytest.raises(ValueError): model(x, a, valid)
