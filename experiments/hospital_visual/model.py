"""Original Uni-AdaFocus only, with complete official loading and new FSN heads."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'models/Uni-AdaFocus-TSM-original'
WEIGHT_SHA = '2dea5c15ce23b3549aeab977774649f0ce8dcbc5637d5d1d1319efc019896fc3'
VARIANT = 'uni_adafocus_original'
CONFIG = {
    'variant': VARIANT, 'official_model': 'Uni-AdaFocus-TSM SSv2 p128 8+12',
    'glance_frames': 8, 'focus_candidates': 36, 'focus_selected': 12,
    'patch_size': 128, 'classes': 7, 'seed': 42,
    'head_warmup_epochs': 5, 'finetune_epochs': 50, 'early_stopping': None,
    'batch_size': 4, 'accumulation_steps': 16, 'drop_last': False,
    'head_lr': .001, 'lr': .005, 'weight_decay': .0005, 'momentum': .9,
    'lr_ratios': {'global_CNN': .5, 'aux_fc': .5, 'spatial_policy': .2,
                  'temporal_policy': .2, 'local_CNN': 1.},
    'class_weights': 'sqrt_inverse_train_only', 'clip_grad': 20.,
    'loss': 'official_two_branch_five_CE_plus_0.5_scale_regularization',
    'precision': 'bf16_autocast_fp32_parameters', 'augmentation': 'seeded_flip_only',
    'selection': 'strict_finetune_heldout_development_macro_f1_same_checkpoint_accuracy',
    'official_full_training_reproduction': False, 'additional_test_read': False,
    'optimizer_steps_per_epoch': 115, 'maximum_optimizer_steps': 6325,
    'following_fixed_T': 'asformer_interval_self_reference',
    'temporal_weights_reused': False,
}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def original_class():
    # The historical wrapper imports the modified tree. Never use it here.
    for name, module in tuple(sys.modules.items()):
        if name == 'ops' or name.startswith(('ops.', 'archs.')) or name == 'archs':
            file = getattr(module, '__file__', None)
            if file and not Path(file).resolve().is_relative_to(SOURCE.resolve()):
                raise ValueError('mixed original/modified AdaFocus imports rejected')
    sys.path.insert(0, str(SOURCE)) if str(SOURCE) not in sys.path else None
    cls = importlib.import_module('archs.uni_adafocus_tsm').AdaFocus
    if not Path(sys.modules[cls.__module__].__file__).resolve().is_relative_to(SOURCE.resolve()):
        raise ValueError('only the preserved Original source is permitted')
    return cls

def arguments(device):
    return SimpleNamespace(num_glance_segments=8, num_input_focus_segments=36,
        num_focus_segments=12, input_size=224, patch_size=128, feature_map_channels=1280,
        device=device, modality='RGB', consensus_type='avg', dropout=.5,
        img_feature_dim=256, no_partialbn=False, pretrain='none', shift=True,
        shift_div=8, shift_place='blockres', tune_from=None, dataset='hospital_fsn',
        temporal_pool=False, non_local=False, stn_hidden_dim=128, temporal_hidden_dim=64)

def permitted_head(name):
    return name in {prefix + suffix for prefix in (
        'global_CNN.new_fc.', 'global_CNN.new_new_fc.',
        'local_CNN.new_fc.', 'local_CNN.new_new_fc.', 'aux_fc.')
        for suffix in ('weight', 'bias')}

def strict_official(core, checkpoint):
    if digest(checkpoint) != WEIGHT_SHA: raise ValueError('official SSv2 bytes changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    source = saved['state_dict']
    state = {}
    for key, value in source.items():
        key = key.removeprefix('module.')
        if key in state or not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ValueError('duplicate or nonfinite official tensor')
        state[key] = value
    current = core.state_dict()
    if set(state) != set(current) or len(state) != 672:
        raise ValueError('complete original architecture must match all 672 official tensors')
    merged = {}
    for key, target in current.items():
        if permitted_head(key):
            shape = (174, target.shape[1]) if target.ndim == 2 else (174,)
            if tuple(state[key].shape) != shape or target.shape[0] != 7:
                raise ValueError('only official 174→7 classifier replacement allowed')
            merged[key] = target
        else:
            if state[key].shape != target.shape: raise ValueError('non-head official tensor shape differs')
            merged[key] = state[key]
    core.load_state_dict(merged, strict=True)
    return {'official_sha256': WEIGHT_SHA, 'all_source_tensors_validated': 672,
            'shared_tensors_loaded': 662, 'only_seven_class_head_tensors_reset': 10,
            'strict_merged_state_load': True, 'old_FSN_weights_loaded': False}

class OriginalVisual(nn.Module):
    def __init__(self, device, checkpoint=None):
        super().__init__()
        self.core = original_class()(7, arguments(device))
        self.load_report = strict_official(self.core, checkpoint) if checkpoint else None

    def forward(self, frames):
        if frames.ndim != 5 or tuple(frames.shape[1:]) != (44, 3, 224, 224):
            raise ValueError('separately sampled RGB8 glance + RGB36 candidates required')
        return self.core(frames[:, :8].contiguous().reshape(-1, 24, 224, 224),
                         frames[:, 8:].contiguous().reshape(-1, 108, 224, 224))

    def loss(self, outputs, target, weights):
        p1, p2 = outputs
        ce = sum(F.cross_entropy(p[j], target, weight=weights) for p in (p1, p2) for j in range(5)) / 2
        regularizer = .5 * (p1[6][:, 2:4] - 1).square().mean()
        return ce + regularizer

def positive_gradients(model):
    groups = {k: 0. for k in CONFIG['lr_ratios']}
    for name, p in model.core.named_parameters():
        if not p.requires_grad: continue
        if p.grad is None or not bool(torch.isfinite(p.grad).all()):
            raise ValueError('missing/nonfinite trainable original gradient')
        groups[name.split('.')[0]] += float(p.grad.detach().float().square().sum())
    return {k: v ** .5 for k, v in groups.items()}

def optimizer(model, head_only=False):
    model.train()  # Apply the official partial-BN freeze before coverage audit.
    if head_only:
        for name, p in model.core.named_parameters(): p.requires_grad_(permitted_head(name))
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
    groups, seen = [], set()
    for group, ratio in CONFIG['lr_ratios'].items():
        parameters = [p for p in getattr(model.core, group).parameters() if p.requires_grad]
        if any(id(p) in seen for p in parameters): raise ValueError('duplicate optimizer parameter')
        seen.update(id(p) for p in parameters)
        groups.append({'params': parameters, 'lr': .005 * ratio, 'weight_decay': .0005, 'name': group})
    if seen != {id(p) for p in model.parameters() if p.requires_grad}: raise ValueError('incomplete optimizer')
    return torch.optim.SGD(groups, lr=.005, momentum=.9)
