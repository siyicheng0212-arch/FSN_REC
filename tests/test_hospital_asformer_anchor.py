from copy import deepcopy

import pytest
import torch

from experiments.hospital_asformer_anchor.model import (
    AnchoredTemporalHead, SelfReferencedCalibration, relative_bias,
    CONFIG, VARIANT, objective, plateau, preparation_status,
)
from experiments.hospital_asformer_interval.model import (
    CONFIG as INTERVAL_CONFIG, IntervalTemporalHead,
)
from experiments.hospital_temporal_compare.model import TemporalHead

torch.set_num_threads(2)


def batch():
    generator = torch.Generator().manual_seed(817)
    x = torch.randn(4, 6, 1543, generator=generator)
    labels = torch.randint(0, 7, (4, 6), generator=generator)
    valid = torch.arange(6)[None] < torch.tensor([1, 2, 4, 6])[:, None]
    start = torch.arange(6).double() * 5
    times = torch.stack((start, start + torch.arange(6).double() + 1), -1)
    return x, x[..., -7:].clone(), labels, valid, times[None].repeat(4, 1, 1)


def nonneutral(model):
    with torch.no_grad():
        model.calibration.network[-1].weight.copy_(torch.linspace(-.2, .2, 16)[None])
    return model


def test_one_calibration_change_only_original_base_rng_and_recipe_preserved():
    old = TemporalHead('asformer')
    interval = IntervalTemporalHead()
    new = AnchoredTemporalHead()
    counts = new.parameter_counts()
    assert counts['original_ASFormer'] == 1096732
    assert counts['calibration'] == 112
    assert counts['trainable'] == 1096844
    assert counts['A_trainable'] == 0
    assert counts['parameter_compute_matched'] is False
    assert new.calibration.network[2].bias is None
    assert new.variant == VARIANT
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key])
    for key in ('weight', 'bias'):
        assert torch.equal(getattr(interval.calibration.network[0], key),
                           getattr(new.calibration.network[0], key))
    for key in ('training', 'runtime_precision', 'multistage_loss', 'asformer',
                'input', 'output', 'early_stopping'):
        assert CONFIG[key] == INTERVAL_CONFIG[key]
    assert CONFIG['selection_order'][:-1] == INTERVAL_CONFIG['selection_order']
    assert CONFIG['selection_order'][-1] == VARIANT
    assert CONFIG['early_stopping']['patience'] == 10
    torch.manual_seed(77)
    before = torch.get_rng_state()
    AnchoredTemporalHead()
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('training', [False, True])
def test_neutral_all_stages_matches_original_with_same_dropout_seed(training):
    x, a, _, valid, times = batch()
    old, new = TemporalHead('asformer'), AnchoredTemporalHead()
    old.train(training); new.train(training)
    torch.manual_seed(51)
    original = old(x, a, valid, all_stages=True)
    torch.manual_seed(51)
    changed = new(x, a, valid, times, all_stages=True)
    torch.testing.assert_close(original, changed, atol=2e-5, rtol=2e-5)


def test_all_negative_scores_can_give_positive_and_negative_relative_bias():
    # Negative last-layer weights are no longer a mathematical guarantee that
    # every neighbor is suppressed. This does not guarantee learned improvement.
    calibration = SelfReferencedCalibration()
    with torch.no_grad():
        calibration.network[0].weight.zero_()
        calibration.network[0].bias.zero_()
        calibration.network[0].weight[0, 0] = 1
        calibration.network[0].bias[0] = 2
        calibration.network[2].weight.fill_(-.2)
    times = torch.tensor([[[0., 1.], [2., 3.], [4., 5.]]], dtype=torch.float64)
    hidden = torch.ones(1, 64, 3)
    probability = torch.full((1, 7, 3), 1 / 7)
    descriptors = calibration.inputs(times, hidden, probability)
    raw = calibration.network(descriptors).squeeze(-1)
    assert bool((raw < 0).all())
    bias = calibration(times, hidden, probability)
    assert bias[0, 1, 0] > 0
    assert bias[0, 1, 2] < 0
    assert torch.count_nonzero(bias.diagonal(dim1=-2, dim2=-1)) == 0
    assert bool((bias.abs() <= 1).all())


def test_shared_score_offsets_cancel_not_just_global_offsets():
    raw = torch.tensor([[[-1., -2., -.5], [-3., -2., -1.], [-.25, -2., -1.]]])
    shift = torch.tensor([[[3.], [-5.], [4.]]])
    torch.testing.assert_close(relative_bias(raw), relative_bias(raw + shift), atol=0, rtol=0)
    expected = torch.tanh(raw - raw.diagonal(dim1=-2, dim2=-1).unsqueeze(-1))
    torch.testing.assert_close(relative_bias(raw), expected, atol=0, rtol=0)


def test_fixed_bias_bound_limits_neighbor_self_odds_not_absolute_probability():
    raw = torch.tensor([[[0., 100., -100.], [-100., 0., 100.], [100., -100., 0.]]])
    bias = relative_bias(raw)
    assert bool((bias.abs() <= 1).all())
    logits = torch.tensor([[[1., -1., 2.], [3., 1., 0.], [-2., 0., 1.]]])
    original, calibrated = logits.softmax(-1), (logits + bias).softmax(-1)
    self_original = original.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    self_calibrated = calibrated.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)
    odds_multiplier = (calibrated / self_calibrated) / (original / self_original)
    torch.testing.assert_close(odds_multiplier, bias.exp())


@pytest.mark.parametrize('length', [1, 2, 3, 5, 8, 17])
def test_nonneutral_mixed_individual_parity_and_no_cross_segment_context(length):
    generator = torch.Generator().manual_seed(819)
    n = max(2, length)
    x = torch.randn(2, n, 1543, generator=generator)
    valid = torch.arange(n)[None] < torch.tensor([1, length])[:, None]
    start = torch.arange(n).double() * 4
    times = torch.stack((start, start + 1), -1)[None].repeat(2, 1, 1)
    model = nonneutral(AnchoredTemporalHead()).eval()
    with torch.no_grad():
        mixed = model(x, x[..., -7:], valid, times, all_stages=True)
        for index, count in enumerate((1, length)):
            one = model(x[index:index+1, :count], x[index:index+1, :count, -7:],
                        valid[index:index+1, :count], times[index:index+1, :count], all_stages=True)
            torch.testing.assert_close(mixed[:, index, :count], one[:, 0], atol=2e-5, rtol=2e-5)
        altered = x.clone(); altered[1] += 1000
        new_times = times.clone(); new_times[1] += 100000
        independent = model(altered, altered[..., -7:], valid, new_times, all_stages=True)
        torch.testing.assert_close(independent[:, 0, :1], mixed[:, 0, :1], atol=2e-5, rtol=2e-5)


def test_singleton_and_center_only_have_no_fabricated_neighbors_after_learning():
    x, a, _, valid, times = batch()
    model = nonneutral(AnchoredTemporalHead()).eval()
    old = TemporalHead('asformer').eval()
    with torch.no_grad():
        new = model(x[:1, :1], a[:1, :1], valid[:1, :1], times[:1, :1])
        torch.testing.assert_close(new, old(x[:1, :1], a[:1, :1], valid[:1, :1]), atol=2e-5, rtol=2e-5)
        isolated = model(x, a, valid, times, mode='center_only')
        expected = old(x, a, valid, mode='center_only')
        torch.testing.assert_close(isolated[valid], expected[valid], atol=2e-5, rtol=2e-5)


def test_three_cpu_steps_actual_learning_base_stage_grads_frozen_A_and_times():
    x, a, labels, valid, times = batch()
    x.requires_grad_(); a.requires_grad_(); times.requires_grad_()
    model = AnchoredTemporalHead()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    positive = {'calibration.network.0.': False, 'calibration.network.2.': False}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective(model(x, a, valid, times, all_stages=True), labels, valid)
        loss.backward()
        assert bool(torch.isfinite(loss))
        assert x.grad is None and a.grad is None and times.grad is None
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        for prefix in ('head.encoder.', 'head.decoders.0.', 'head.decoders.1.', 'head.decoders.2.'):
            assert sum(float(p.grad.square().sum()) for name, p in model.named_parameters()
                       if name.startswith(prefix)) > 0
        for prefix in positive:
            positive[prefix] |= any(bool(p.grad.abs().any()) for name, p in model.named_parameters()
                                    if name.startswith(prefix))
        if step == 0:
            assert torch.count_nonzero(model.calibration.network[0].weight.grad) == 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
    assert all(positive.values())
    for prefix in (*positive, 'head.encoder.', 'head.decoders.0.', 'head.decoders.1.', 'head.decoders.2.'):
        assert any(not torch.equal(before[name], p) for name, p in model.named_parameters()
                   if name.startswith(prefix))


@pytest.mark.parametrize('fault', ['nan', 'inf', 'empty', 'nonsquare', 'dtype', 'rank'])
def test_bad_scores_rejected(fault):
    scores = torch.zeros(1, 2, 2)
    if fault == 'nan': scores[0, 0, 1] = float('nan')
    elif fault == 'inf': scores[0, 0, 1] = float('inf')
    elif fault == 'empty': scores = torch.zeros(1, 0, 0)
    elif fault == 'nonsquare': scores = torch.zeros(1, 2, 3)
    elif fault == 'dtype': scores = scores.long()
    else: scores = torch.zeros(2, 2)
    with pytest.raises(ValueError): relative_bias(scores)


@pytest.mark.parametrize('fault', ['variant', 'config', 'identity', 'nonfinite', 'old_tensor'])
def test_checkpoint_cannot_mislabel_old_formula_or_wrong_identity(fault):
    model = AnchoredTemporalHead()
    identity = {'synthetic_identity': 'registered'}
    payload = {'variant': VARIANT, 'config': deepcopy(CONFIG), 'identity': identity,
               'model': deepcopy(model.state_dict())}
    if fault == 'variant': payload['variant'] = 'asformer_interval'
    elif fault == 'config': payload['config']['interval_calibration']['fixed_absolute_bias_bound'] = 2
    elif fault == 'identity': payload['identity'] = {'synthetic_identity': 'not_registered'}
    elif fault == 'nonfinite': next(iter(payload['model'].values())).flatten()[0] = float('nan')
    else: payload['model'] = IntervalTemporalHead().state_dict()
    with pytest.raises((ValueError, RuntimeError)): model.load_registered_checkpoint(payload, identity)


def test_checkpoint_roundtrip_and_preparation_not_mislabeled_training():
    original = nonneutral(AnchoredTemporalHead())
    identity = {'synthetic_identity': 'registered'}
    payload = {'variant': VARIANT, 'config': CONFIG, 'identity': identity, 'model': original.state_dict()}
    restored = AnchoredTemporalHead().load_registered_checkpoint(payload, identity)
    for name, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
    assert preparation_status()['model_implemented'] is True
    assert preparation_status()['execution_pipeline_migrated'] is True
    for key in ('server_deployed', 'CUDA_acceptance_completed', 'formal_training_started'):
        assert preparation_status()[key] is False
    rows = [{'epoch': i, 'val_macro_f1': .7} for i in range(1, 12)]
    assert plateau(rows)['should_stop'] is True


def test_clock_translation_padding_values_and_raw_logit_pairing_preserved():
    x, a, _, valid, times = batch()
    model = nonneutral(AnchoredTemporalHead()).eval()
    with torch.no_grad():
        reference = model(x, a, valid, times, all_stages=True)
        translated = model(x, a, valid, times + 1e6, all_stages=True)
        torch.testing.assert_close(reference, translated, atol=2e-5, rtol=2e-5)
        changed_x, changed_times = x.clone(), times.clone()
        changed_x[~valid] += 1e4
        changed_times[~valid] = torch.tensor([1e8, 1e8 + 1], dtype=torch.float64)
        padded = model(changed_x, a, valid, changed_times, all_stages=True)
        torch.testing.assert_close(reference[:, valid], padded[:, valid], atol=2e-5, rtol=2e-5)
    bad = a.clone(); bad[0, 0, 0] += 1
    with pytest.raises(ValueError): model(x, bad, valid, times)
    with pytest.raises(ValueError): model(x, a, valid, times.float())
    gap_mask = valid.clone(); gap_mask[1] = torch.tensor([1, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError): model(x, a, gap_mask, times)


