"""Only decoder attention logits change; original encoder, values and loss remain."""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from experiments.hospital_temporal_compare.model import CONFIG as BASE, TemporalHead, objective

VARIANT = "asformer_interval"
CONFIG = copy.deepcopy(BASE)
CONFIG.update(variants=[VARIANT], selection_order=[*BASE["selection_order"], "mstcnpp_visual_r", VARIANT],
    interval_calibration={"pair_inputs": ["signed_log1p_midpoint_distance_seconds",
        "log_neighbor_to_anchor_duration_ratio", "previous_stage_hidden_cosine",
        "previous_stage_anchor_normalized_entropy", "previous_stage_neighbor_normalized_entropy"],
        "time_unit_seconds": 1.0, "mlp": [5,16,1], "activation": "ReLU",
        "sharing": "one_network_all_30_decoder_attention_layers", "zero_initialized_output": True,
        "self_bias": 0.0, "virtual_padding_bias": 0.0, "original_windows_unchanged": True,
        "intervals": "sealed_manifest_clip_start_sec_clip_end_sec_not_frame_indices",
        "previous_hidden_is_indirect_visual_evidence_not_raw_A_visual": True,
        "no_new_loss_gate_residual_or_label_input": True},
    early_stopping={"criterion":"Menzhen_development_val_macro_f1", "patience":10,
        "min_delta":0, "exact_tie_is_non_improvement":True, "maximum_epochs":100,
        "activation":"from_first_epoch"})
CONFIG["training"]["early_stopping"] = True


class PairCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        # Added construction must not consume the original model/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.network = nn.Sequential(nn.Linear(5,16), nn.ReLU(), nn.Linear(16,1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    @staticmethod
    def inputs(intervals, hidden, probabilities):
        if intervals.ndim!=3 or intervals.shape[-1]!=2 or hidden.shape[:1]!=intervals.shape[:1] \
                or hidden.shape[-1]!=intervals.shape[1] or probabilities.shape!=(len(intervals),7,intervals.shape[1]) \
                or not bool(torch.isfinite(intervals).all() and torch.isfinite(hidden).all()
                            and torch.isfinite(probabilities).all()) \
                or not bool((intervals[...,1]>intervals[...,0]).all()):
            raise ValueError("finite positive-duration real intervals and same-stage evidence required")
        # Float64 subtraction before float32 avoids recording-clock precision loss.
        duration = intervals[...,1]-intervals[...,0]
        mid = intervals[...,0]+duration/2
        distance = mid[:,None,:]-mid[:,:,None]
        signed = distance.sign()*torch.log1p(distance.abs())
        ratio = torch.log(duration[:,None,:]/duration[:,:,None])
        normalized = F.normalize(hidden.transpose(1,2),dim=-1,eps=1e-12)
        cosine = torch.bmm(normalized, normalized.transpose(1,2))
        p = probabilities.clamp_min(1e-12)
        entropy = -(probabilities*p.log()).sum(1)/math.log(7)
        n = intervals.shape[1]
        return torch.stack((signed.to(hidden.dtype), ratio.to(hidden.dtype), cosine,
            entropy[:,:,None].expand(-1,-1,n),entropy[:,None,:].expand(-1,n,-1)),dim=-1)

    def forward(self, intervals, hidden, probabilities):
        features = self.inputs(intervals,hidden,probabilities)
        bias = self.network(features).squeeze(-1)
        # A real singleton has no context pair to calibrate. No fabricated times
        # or bias for virtual padding; do not alter the author's softmax mass.
        diagonal = torch.eye(bias.shape[-1],device=bias.device,dtype=torch.bool)[None]
        return bias.masked_fill(diagonal,0)


def calibrated_attention(layer, x, previous_hidden, mask, bias):
    """Original sliding attention with one additive real-pair score correction."""
    if layer.stage!="decoder" or layer.att_type!="sliding_att":
        raise ValueError("only registered decoder sliding attention may be calibrated")
    q,k,v = layer.query_conv(x),layer.key_conv(x),layer.value_conv(previous_hidden)
    batch, channels, length = q.shape
    if bias.shape!=(batch,length,length) or not bool(torch.isfinite(bias).all()):
        raise ValueError("finite same-segment pair bias required")
    bl=layer.bl
    if layer.fast_short_windows and bl>=2*length:
        energy=torch.bmm(q.transpose(1,2),k)/math.sqrt(channels)
        energy=energy+math.log(1+1e-6)+bias
        missing=bl+2*(bl//2)-length
        extra=energy.new_full((batch,length,1),math.log(missing*1e-6))
        attention=F.softmax(torch.cat((energy,extra),-1),-1)[...,:length]
        output=torch.bmm(v,attention.transpose(1,2))
        return layer.conv_out(F.relu(output))*mask[:,0:1]
    nb=math.ceil(length/bl); padded=nb*bl; bank=bl+2*(bl//2)
    q=F.pad(q,(0,padded-length)); k=F.pad(k,(bl//2,padded-length+bl//2))
    v=F.pad(v,(bl//2,padded-length+bl//2))
    padmask=F.pad(mask[:,0:1],(bl//2,padded-length+bl//2))
    query=q.reshape(batch,channels,nb,bl).permute(0,2,1,3).reshape(batch*nb,channels,bl)
    key=torch.stack([k[:,:,i*bl:i*bl+bank] for i in range(nb)],1).reshape(batch*nb,channels,bank)
    val=torch.stack([v[:,:,i*bl:i*bl+bank] for i in range(nb)],1).reshape(batch*nb,v.shape[1],bank)
    pm=torch.stack([padmask[:,:,i*bl:i*bl+bank] for i in range(nb)],1).reshape(batch*nb,1,bank)
    final_mask=layer.window_mask.repeat(batch*nb,1,1)*pm
    qi=torch.arange(padded,device=x.device).reshape(nb,bl)
    ki=torch.arange(bank,device=x.device)[None]+torch.arange(nb,device=x.device)[:,None]*bl-bl//2
    real=(qi[:,:,None]<length)&(ki[:,None,:]>=0)&(ki[:,None,:]<length)
    local=bias[:,qi.clamp_max(length-1)[:,:,None],ki.clamp(0,length-1)[:,None,:]]
    local=local.masked_fill(~real[None],0).reshape(batch*nb,bl,bank)
    energy=torch.bmm(query.transpose(1,2),key)/math.sqrt(channels)
    attention=F.softmax(energy+torch.log(final_mask+1e-6)+local*final_mask,-1)*final_mask
    output=torch.bmm(val,attention.transpose(1,2))
    output=layer.conv_out(F.relu(output))
    output=output.reshape(batch,nb,-1,bl).permute(0,2,1,3).reshape(batch,-1,padded)
    return output[:,:,:length]*mask[:,0:1]


class IntervalTemporalHead(TemporalHead):
    def __init__(self):
        super().__init__("asformer")
        self.variant=VARIANT
        self.calibration=PairCalibration()

    def sequence(self, x, intervals):
        mask=x.new_ones((len(x),1,x.shape[1]))
        out,previous=self.head.encoder(x.transpose(1,2),mask)
        outputs=[out]
        for decoder in self.head.decoders:
            probability=F.softmax(out,dim=1)*mask
            bias=self.calibration(intervals,previous,probability)
            feature=decoder.conv_1x1(probability)
            for layer in decoder.layers:
                local=layer.feed_forward(feature)
                normalized=torch.zeros_like(local) if local.shape[-1]==1 else layer.instance_norm(local)
                local=layer.alpha*calibrated_attention(layer.att_layer,normalized,previous,mask,bias)+local
                local=layer.dropout(layer.conv_1x1(local))
                feature=(feature+local)*mask
            out=decoder.conv_out(feature)*mask
            previous=feature
            outputs.append(out)
        return torch.stack(outputs)

    def forward(self, features, a_logits, valid, intervals, mode="native", *, all_stages=False):
        if mode not in {"native","center_only"} or features.ndim!=3 or features.shape[-1]!=1543 \
                or a_logits.shape!=(*features.shape[:2],7) or valid.shape!=features.shape[:2] \
                or valid.dtype!=torch.bool or not bool(valid.any(1).all()) \
                or intervals.shape!=(*features.shape[:2],2) or intervals.dtype!=torch.float64 \
                or not bool(torch.isfinite(features).all() and torch.isfinite(a_logits).all()
                            and torch.isfinite(intervals).all()) \
                or not torch.equal(features[...,-7:][valid],a_logits[valid]) \
                or not bool((intervals[...,1][valid]>intervals[...,0][valid]).all()):
            raise ValueError("paired frozen evidence and exact float64 real intervals required")
        lengths=valid.sum(1)
        if not torch.equal(valid,torch.arange(valid.shape[1],device=valid.device)[None]<lengths[:,None]):
            raise ValueError("contiguous left-aligned sealed segments required")
        features=features.detach(); intervals=intervals.detach()
        stages=features.new_zeros((4,*features.shape[:2],7))
        if mode=="center_only":
            isolated=features[valid][:,None]; times=intervals[valid][:,None]
            y=self.forward(isolated,isolated[...,-7:],torch.ones(len(isolated),1,dtype=torch.bool,
                device=features.device),times,all_stages=True)
            stages[:,valid]=y[:,:,0]
        else:
            for length in sorted(set(lengths.tolist())):
                rows=torch.where(lengths==length)[0]
                y=self.sequence(features[rows,:length],intervals[rows,:length])
                stages[:,rows,:length]=y.permute(0,1,3,2)
        if not bool(torch.isfinite(stages).all()): raise ValueError("nonfinite calibrated logits")
        return stages if all_stages else stages[-1]

    def parameter_counts(self):
        return {"trainable":sum(p.numel() for p in self.parameters()),"A_trainable":0,
            "original_ASFormer":sum(p.numel() for p in self.head.parameters()),
            "calibration":sum(p.numel() for p in self.calibration.parameters()),
            "parameter_compute_matched":False}


def plateau(history):
    high=-1.0; stale=0; best_epoch=None
    for epoch,row in enumerate(history,1):
        value=row["val_macro_f1"]
        if row["epoch"]!=epoch or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError("ordered finite completed-epoch validation history required")
        if stale>=10: raise ValueError("new trainer must stop at first authorized plateau")
        if value>high: high,stale,best_epoch=value,0,epoch
        else: stale+=1
    return {"best_epoch":best_epoch,"best_macro_f1":high,"consecutive_no_improvement":stale,
            "should_stop":stale>=10}
