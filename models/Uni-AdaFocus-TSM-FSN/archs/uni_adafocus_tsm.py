import torch.nn.parallel
import torch.optim

from ops.models import TSN
from ops.transforms import *
from archs.fsn_modules import LocalContextInteraction

import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import autocast

best_prec1 = 0


def get_patch_grid_scalexy(input_frames, action, image_size, patch_size, input_patch_size):
    batchsize = action.size(0)
    theta = torch.zeros((batchsize, 2, 3), device=input_frames.device)
    patch_scale = action[:, 2:4] * (224 - 96) / input_patch_size + 96 / input_patch_size
    patch_coordinate = (action[:, :2] * (image_size - patch_size * patch_scale))
    x1, x2, y1, y2 = patch_coordinate[:, 1], patch_coordinate[:, 1] + patch_size * patch_scale[:, 1], \
                     patch_coordinate[:, 0], patch_coordinate[:, 0] + patch_size * patch_scale[:, 0]

    theta[:, 0, 0], theta[:, 1, 1] = patch_size * patch_scale[:, 1] / image_size, patch_size * patch_scale[:, 0] / image_size
    theta[:, 0, 2], theta[:, 1, 2] = -1 + (x1 + x2) / image_size, -1 + (y1 + y2) / image_size

    grid = F.affine_grid(theta.float(), torch.Size((batchsize, 3, patch_size, patch_size)))

    patches = F.grid_sample(input_frames, grid)

    return patches


def cal_x(x1, y1, x2, y2, y):
    x = (x2 * (y - y1) - x1 * (y - y2)) / (y2 - y1)
    return x


def policy_sample_indices(weights, num_segments, num_global_segments, num_local_segments, rand_sample=True, device="cpu"):
    _b, _ = weights.shape
    integrated = weights.cumsum(dim=1)
    new_col = torch.zeros(_b, device=device)
    integrated = torch.cat([new_col.unsqueeze(1), integrated], dim=1)  # Accumulation of weights, with leading 0
    integrated[:, -1] = 1.0  # fix precision loss
    threshold = torch.arange(0, num_local_segments, device=device).float().repeat(_b, 1)
    if rand_sample:
        threshold = (threshold + torch.rand_like(threshold)) / num_local_segments
    else:
        threshold = (threshold + 0.5) / num_local_segments
    # find first integrated[j, ans] > threshold[j, i] - eps
    cmp_integrated_threshold = integrated.unsqueeze(2) >= threshold.unsqueeze(1)
    first_cur = cmp_integrated_threshold.max(dim=1)[1] - 1
    quantile = cal_x(x1=first_cur / num_global_segments, y1=integrated.gather(1, first_cur),
                     x2=(first_cur + 1) / num_global_segments, y2=integrated.gather(1, first_cur + 1),
                     y=threshold)
    # round float to integer index
    real_indices = quantile * num_segments
    int_indices = torch.floor(real_indices).long()
    # fix overlap situation
    for i in range(1, num_local_segments):
        int_indices[:, i] = torch.where(int_indices[:, i].gt(int_indices[:, i - 1]), int_indices[:, i], int_indices[:, i - 1] + 1)
    lim = num_segments - num_local_segments + torch.arange(0, num_local_segments, device=device)
    lim = lim.unsqueeze(0)
    int_indices = torch.where(int_indices.gt(lim), lim, int_indices)
    offset = torch.arange(0, _b, device=device).unsqueeze(1) * num_segments

    int_indices = int_indices + offset
    return int_indices.view(-1)


def MCSampleFeature(global_feat, weights, global_feature_dim, T1, sample_times=1000, device="cpu"):
    B, _ = weights.shape
    global_feat = global_feat.view(B, -1, global_feature_dim)
    _, T, Dim = global_feat.shape
    all_weights = weights
    all_weights = all_weights.unsqueeze(1).repeat(1, sample_times, 1).view(-1, T)
    sum_feat = torch.zeros([B * sample_times, global_feature_dim], device=device).float()
    for i in range(T1):
        new_weights = all_weights.clone()
        if i > 0:
            indices = torch.multinomial(all_weights, i, replacement=False)
            new_weights = new_weights.scatter(dim=1, index=indices, value=0.0)
        new_feat = global_feat.unsqueeze(1).repeat(1, sample_times, 1, 1).view(-1, T, Dim)
        frame_feat = (new_feat * new_weights.unsqueeze(-1)).sum(dim=1)
        frame_feat = frame_feat / new_weights.sum(dim=1).detach().unsqueeze(-1)
        sum_feat = sum_feat + frame_feat
    sum_feat = sum_feat / T1
    sum_feat = sum_feat.view(-1, sample_times, Dim).mean(dim=1)
    return sum_feat


class TemporalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_glance_segments, temperature):
        super().__init__()
        self.input_dim = input_dim
        self.num_glance_segments = num_glance_segments
        self.T = temperature
        self.Tconv = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.input_dim,
                out_channels=hidden_dim,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            )
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(-1, self.num_glance_segments, self.input_dim).permute(0, 2, 1).contiguous().unsqueeze(-1)
        x = self.Tconv(x).flatten(1)
        x1 = self.softmax(x)
        x2 = x / self.T
        x2 = self.softmax(x2)
        return x1, x2


class SpatialPolicy(nn.Module):
    def __init__(self, stn_feature_dim, hidden_dim, num_glance_segments):
        super(SpatialPolicy, self).__init__()
        self.stn_feature_dim = stn_feature_dim
        self.num_glance_segments = num_glance_segments
        self.encoder = nn.Sequential(
            nn.Conv3d(
                stn_feature_dim, hidden_dim,
                kernel_size=(1, 1, 1), stride=1, padding=0, bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=(3, 3, 3), stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=(3, 3, 3), stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(
                hidden_dim, 4,
                kernel_size=(self.num_glance_segments, 7, 7), stride=1, padding=(0, 0, 0), bias=False,
            ),
            nn.Sigmoid()
        )

    def forward(self, features):
        N, TC, H, W = features.shape
        features = features.reshape(
            N, self.num_glance_segments, self.stn_feature_dim, H, W
        ).permute(0, 2, 1, 3, 4).contiguous()
        actions = self.encoder(features).reshape(N, 4)
        return actions


class AdaFocus(nn.Module):
    def __init__(self, num_class, args):
        super(AdaFocus, self).__init__()
        self.num_glance_segments = args.num_glance_segments
        self.num_input_focus_segments = args.num_input_focus_segments
        self.num_focus_segments = args.num_focus_segments
        self.input_size = args.input_size
        self.patch_size = args.patch_size
        self.global_feature_dim = args.feature_map_channels
        self.num_classes = num_class
        self.device = args.device
        self.fsn_local_adapter = getattr(args, 'fsn_local_adapter', 'none')
        self.fsn_interaction_mode = getattr(args, 'fsn_interaction', 'none')
        self.fsn_interaction_weight = getattr(args, 'fsn_interaction_weight', 0.2)
        self.fsn_local_grid_size = getattr(args, 'fsn_local_grid_size', 3)
        self.use_fsn_local_features = (
            self.fsn_local_adapter != 'none' or self.fsn_interaction_mode != 'none'
        )

        self.global_CNN = TSN(num_class, self.num_glance_segments, args.modality,
                              base_model='mobilenetv2',
                              consensus_type=args.consensus_type,
                              dropout=args.dropout,
                              img_feature_dim=args.img_feature_dim,
                              partial_bn=not args.no_partialbn,
                              pretrain=args.pretrain,
                              is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                              fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                              temporal_pool=args.temporal_pool,
                              non_local=args.non_local)
        self.local_CNN = TSN(num_class, self.num_focus_segments, args.modality,
                             base_model='resnet50',
                             consensus_type=args.consensus_type,
                             dropout=args.dropout,
                             img_feature_dim=args.img_feature_dim,
                             partial_bn=not args.no_partialbn,
                             pretrain=args.pretrain,
                             is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                             fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                             temporal_pool=args.temporal_pool,
                             non_local=args.non_local,
                             return_feature_grid=self.use_fsn_local_features,
                             local_adapter_mode=self.fsn_local_adapter,
                             local_adapter_dim=getattr(args, 'fsn_adapter_dim', 256),
                             local_grid_size=self.fsn_local_grid_size)
        self.aux_fc = nn.Linear(self.global_feature_dim, num_class)
        self.spatial_policy = SpatialPolicy(
            stn_feature_dim=args.feature_map_channels,
            hidden_dim=args.stn_hidden_dim,
            num_glance_segments=self.num_glance_segments
            )
        self.temporal_policy = TemporalPolicy(
            input_dim=self.global_feature_dim,
            hidden_dim=args.temporal_hidden_dim,
            num_glance_segments=self.num_glance_segments,
            temperature=1.0
        )
        if self.fsn_interaction_mode != 'none':
            self.fsn_interaction = LocalContextInteraction(
                local_channels=2048,
                global_channels=self.global_feature_dim,
                dim=getattr(args, 'fsn_interaction_dim', 256),
                num_classes=num_class,
                mode=self.fsn_interaction_mode,
                heads=getattr(args, 'fsn_interaction_heads', 4),
                dropout=getattr(args, 'fsn_interaction_dropout', 0.1),
                local_grid_size=self.fsn_local_grid_size,
                global_grid_size=getattr(args, 'fsn_global_grid_size', 3),
            )
        else:
            self.fsn_interaction = None

    @staticmethod
    def _uniform_positions(batch, time, device):
        positions = (torch.arange(time, device=device).float() + 0.5) / time
        return positions.unsqueeze(0).expand(batch, -1)

    def _prepare_time_positions(self, global_feat_maps, focus_indices, glance_positions, input_positions):
        batch = global_feat_maps.shape[0]
        if glance_positions is None:
            glance_positions = self._uniform_positions(
                batch, self.num_glance_segments, global_feat_maps.device
            )
        else:
            glance_positions = glance_positions.to(global_feat_maps.device).float()
        if input_positions is None:
            input_positions = self._uniform_positions(
                batch, self.num_input_focus_segments, global_feat_maps.device
            )
        else:
            input_positions = input_positions.to(global_feat_maps.device).float()
        focus_positions = input_positions.reshape(-1)[focus_indices]
        focus_positions = focus_positions.view(batch, self.num_focus_segments)
        return glance_positions, focus_positions

    def _interaction_logits(self, local_grid, global_grid, local_positions, global_positions):
        if self.fsn_interaction is None:
            return None
        logits, _ = self.fsn_interaction(
            local_grid, global_grid, local_positions, global_positions
        )
        return logits

    def forward(self, images_glance, images_input, glance_positions=None, input_positions=None):
        global_final_logit, global_avg_logit, global_feat_maps, global_feat, global_logit = self.global_CNN(images_glance, glance=True)
        batch = global_final_logit.shape[0]
        global_grid = global_feat_maps.view(
            batch,
            self.num_glance_segments,
            self.global_feature_dim,
            global_feat_maps.shape[-2],
            global_feat_maps.shape[-1],
        )
        with autocast(enabled=False):
            B, T0, _ = global_logit.shape
            weights, weights_T = self.temporal_policy(global_feat.detach().float())
            temporal_sample_logits = MCSampleFeature(
                global_logit.view(-1, self.num_classes).detach().float(),
                weights,
                global_feature_dim=self.num_classes,
                T1=self.num_glance_segments // 3,
                sample_times=128,
                device=self.device)

            action_3 = self.spatial_policy(global_feat_maps.detach().float())
            patches_3 = get_patch_grid_scalexy(
                input_frames=global_feat_maps.detach().float(),
                action=action_3,
                image_size=7,
                patch_size=self.patch_size // 32,
                input_patch_size=self.patch_size
            )
            aux_global_feat_3 = patches_3.view(-1, T0, self.global_feature_dim, self.patch_size // 32,
                                               self.patch_size // 32)
            aux_global_feat_3 = aux_global_feat_3.mean(dim=[1, 3, 4])
            spatial_sample_logits_3 = self.aux_fc(aux_global_feat_3)

        if self.training:
            with autocast(enabled=False):
                focus_indices = policy_sample_indices(weights_T.detach(), self.num_input_focus_segments,
                                                                self.num_glance_segments,
                                                                self.num_focus_segments, True, self.device)
                images_focus = images_input.view(-1, 3, self.input_size, self.input_size)[focus_indices]
                images_focus = images_focus.view(-1, self.num_focus_segments * 3, self.input_size, self.input_size)
                action_2 = self.spatial_policy(global_feat_maps.detach().float())
                action_1 = torch.rand_like(action_2)
                action_1[:, 2:] = (self.patch_size - 96) / (224 - 96)

                patches_1 = get_patch_grid_scalexy(
                    input_frames=images_focus,
                    action=action_1.detach(),
                    image_size=images_focus.size(2),
                    patch_size=self.patch_size,
                    input_patch_size=self.patch_size
                )
                patches_2 = get_patch_grid_scalexy(
                    input_frames=images_focus,
                    action=action_2.detach(),
                    image_size=images_focus.size(2),
                    patch_size=self.patch_size,
                    input_patch_size=self.patch_size
                )

            B = action_1.size(0)
            local_outputs = self.local_CNN(torch.cat([patches_1, patches_2], dim=0))
            l_1_temp, l_2_temp = local_outputs[:2]
            local_final_logit_1, local_final_logit_2 = l_1_temp[:B], l_1_temp[B:]
            local_avg_logit_1, local_avg_logit_2 = l_2_temp[:B], l_2_temp[B:]

            interaction_logit_1 = interaction_logit_2 = None
            if self.use_fsn_local_features:
                local_grid = local_outputs[2]
                local_grid_1, local_grid_2 = local_grid[:B], local_grid[B:]
                global_positions, local_positions = self._prepare_time_positions(
                    global_feat_maps, focus_indices, glance_positions, input_positions
                )
                interaction_logit_1 = self._interaction_logits(
                    local_grid_1, global_grid, local_positions, global_positions
                )
                interaction_logit_2 = self._interaction_logits(
                    local_grid_2, global_grid, local_positions, global_positions
                )

            final_logit_1 = global_final_logit + local_final_logit_1
            final_logit_2 = global_final_logit + local_final_logit_2
            if interaction_logit_1 is not None:
                final_logit_1 = final_logit_1 + self.fsn_interaction_weight * interaction_logit_1
                final_logit_2 = final_logit_2 + self.fsn_interaction_weight * interaction_logit_2

            return (final_logit_1, global_avg_logit, local_avg_logit_1, temporal_sample_logits, spatial_sample_logits_3, action_1, action_3, weights, focus_indices), \
                   (final_logit_2, global_avg_logit, local_avg_logit_2, temporal_sample_logits, spatial_sample_logits_3, action_2, action_3, weights, focus_indices)

        else:
            focus_indices = policy_sample_indices(weights_T.detach(), self.num_input_focus_segments,
                                                               self.num_glance_segments,
                                                               self.num_focus_segments, False, self.device)
            images_focus = images_input.view(-1, 3, self.input_size, self.input_size)[focus_indices]
            images_focus = images_focus.view(-1, self.num_focus_segments * 3, self.input_size, self.input_size)

            action = self.spatial_policy(global_feat_maps.detach())
            patches = get_patch_grid_scalexy(
                input_frames=images_focus,
                action=action,
                image_size=images_focus.size(2),
                patch_size=self.patch_size,
                input_patch_size=self.patch_size
            )
            local_outputs = self.local_CNN(patches)
            local_final_logit, local_avg_logit = local_outputs[:2]
            final_logit = global_final_logit + local_final_logit
            if self.use_fsn_local_features:
                global_positions, local_positions = self._prepare_time_positions(
                    global_feat_maps, focus_indices, glance_positions, input_positions
                )
                interaction_logit = self._interaction_logits(
                    local_outputs[2], global_grid, local_positions, global_positions
                )
                if interaction_logit is not None:
                    final_logit = final_logit + self.fsn_interaction_weight * interaction_logit
            return final_logit, global_avg_logit, local_avg_logit, temporal_sample_logits, spatial_sample_logits_3, action, action_3, weights, focus_indices

    def get_optim_policies(self, args):
        policies = [{'params': self.temporal_policy.parameters(), 'lr_mult': args.temporal_lr_ratio, 'decay_mult': 1, 'name': "temporal_policy"}] \
               + [{'params': self.spatial_policy.parameters(), 'lr_mult': args.stn_lr_ratio, 'decay_mult': 1, 'name': "spatial_policy"}] \
               + [{'params': self.global_CNN.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1, 'name': "global_CNN"}] \
               + [{'params': self.aux_fc.parameters(), 'lr_mult': args.global_lr_ratio, 'decay_mult': 1, 'name': "aux_fc"}] \
               + [{'params': self.local_CNN.parameters(), 'lr_mult': 1, 'decay_mult': 1, 'name': "local_CNN"}]
        if self.fsn_interaction is not None:
            policies.append({
                'params': self.fsn_interaction.parameters(),
                'lr_mult': getattr(args, 'fsn_module_lr_ratio', 1.0),
                'decay_mult': 1,
                'name': 'fsn_interaction',
            })
        return policies
