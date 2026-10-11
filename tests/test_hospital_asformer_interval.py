from copy import deepcopy
import pytest
import torch
from experiments.hospital_asformer_interval.model import (IntervalTemporalHead, PairCalibration,
    calibrated_attention, plateau, CONFIG, objective)
from experiments.hospital_temporal_compare.model import TemporalHead, CONFIG as BASE

torch.set_num_threads(2)

def batch():
    torch.manual_seed(817)
    x=torch.randn(4,6,1543); y=torch.randint(0,7,(4,6))
    valid=torch.arange(6)[None]<torch.tensor([1,2,4,6])[:,None]
    times=torch.stack((torch.arange(6)*4,torch.arange(6)*4+torch.arange(6)+1),-1).double()
    times=times[None].repeat(4,1,1)
    return x,x[...,-7:].clone(),y,valid,times

def nonneutral(model):
    with torch.no_grad(): model.calibration.network[-1].weight.fill_(.1)
    return model

def test_only_one_bias_network_added_original_weights_rng_and_recipe_equal():
    old,new=TemporalHead('asformer'),IntervalTemporalHead()
    assert new.parameter_counts()['original_ASFormer']==1096732
    assert new.parameter_counts()['calibration']==113
    assert new.parameter_counts()['trainable']==1096845
    for key,value in old.state_dict().items(): assert torch.equal(value,new.state_dict()[key])
    for key in ('runtime_precision','multistage_loss','asformer','input','output'):
        assert CONFIG[key]==BASE[key]
    assert {k:v for k,v in CONFIG['training'].items() if k!='early_stopping'}=={k:v for k,v in BASE['training'].items() if k!='early_stopping'}
    assert CONFIG['early_stopping']['patience']==10
    torch.manual_seed(77); before=torch.get_rng_state(); IntervalTemporalHead()
    assert torch.equal(before,torch.get_rng_state())

def test_initial_all_stages_equal_base_for_variable_length_and_training_dropout():
    x,a,_,v,t=batch(); old=TemporalHead('asformer'); new=IntervalTemporalHead()
    for training in (False,True):
        old.train(training);new.train(training)
        torch.manual_seed(51); b=old(x,a,v,all_stages=True)
        torch.manual_seed(51); c=new(x,a,v,t,all_stages=True)
        torch.testing.assert_close(b,c,atol=2e-5,rtol=2e-5)

@pytest.mark.parametrize('length',[1,2,3,5,8,17])
def test_each_attention_window_neutral_matches_original_including_virtual_mass(length):
    model=IntervalTemporalHead();x=torch.randn(2,64,length);f=torch.randn_like(x);mask=torch.ones(2,1,length)
    bias=torch.zeros(2,length,length)
    for block in model.head.decoders[0].layers:
        layer=block.att_layer
        torch.testing.assert_close(layer(x,f,mask),calibrated_attention(layer,x,f,mask,bias),atol=2e-5,rtol=2e-5)

def test_nonneutral_all_stages_mixed_independent_and_padding_is_ignored():
    x,a,_,v,t=batch(); model=nonneutral(IntervalTemporalHead()).eval()
    with torch.no_grad():
        mixed=model(x,a,v,t,all_stages=True)
        for i,n in enumerate((1,2,4,6)):
            one=model(x[i:i+1,:n],a[i:i+1,:n],v[i:i+1,:n],t[i:i+1,:n],all_stages=True)
            torch.testing.assert_close(mixed[:,i,:n],one[:,0],atol=2e-5,rtol=2e-5)
        xx=x.clone();tt=t.clone();xx[0,1:]+=1000;tt[0,1:]+=10000
        torch.testing.assert_close(model(xx,a,v,tt,all_stages=True)[:,0,:1],mixed[:,0,:1])

def test_singleton_equals_base_even_after_bias_training_and_self_bias_is_zero():
    x,a,_,v,t=batch();m=nonneutral(IntervalTemporalHead()).eval();old=TemporalHead('asformer').eval()
    torch.testing.assert_close(m(x[:1,:1],a[:1,:1],v[:1,:1],t[:1,:1]),old(x[:1,:1],a[:1,:1],v[:1,:1]))
    b=m.calibration(t[:1],torch.randn(1,64,6),torch.softmax(torch.randn(1,7,6),1))
    assert torch.count_nonzero(b.diagonal(dim1=-2,dim2=-1))==0

def test_pair_inputs_exact_time_ratio_entropy_cosine_and_clock_translation():
    t=torch.tensor([[[0.,2.],[3.,7.]]],dtype=torch.float64)
    h=torch.ones(1,64,2);p=torch.full((1,7,2),1/7)
    f=PairCalibration.inputs(t,h,p)
    assert f.shape==(1,2,2,5)
    torch.testing.assert_close(f[0,0,1],torch.tensor([torch.log1p(torch.tensor(4.)).item(),
        torch.log(torch.tensor(2.)).item(),1.,1.,1.]))
    torch.testing.assert_close(f,PairCalibration.inputs(t+1000000,h,p))
    assert f[0,1,0,0]<0 and f[0,1,0,1]<0

def test_frozen_visual_and_interval_detached_stage_and_calibration_updates():
    x,a,y,v,t=batch();x.requires_grad_();a.requires_grad_();t.requires_grad_()
    model=IntervalTemporalHead();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    for step in range(3):
        opt.zero_grad();loss=objective(model(x,a,v,t,all_stages=True),y,v)[0];loss.backward()
        assert x.grad is None and a.grad is None and t.grad is None
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if step==0: assert torch.count_nonzero(model.calibration.network[0].weight.grad)==0
        for prefix in ('head.encoder.','head.decoders.0.','head.decoders.1.','head.decoders.2.'):
            assert sum(float(p.grad.square().sum()) for n,p in model.named_parameters() if n.startswith(prefix))>0
        opt.step()
    assert not torch.equal(before['calibration.network.0.weight'],model.calibration.network[0].weight)
    assert not torch.equal(before['calibration.network.2.weight'],model.calibration.network[2].weight)

@pytest.mark.parametrize('mutation',['zero_duration','negative_duration','nonfinite_time','dtype','nonfinite_feature','logits','mask','empty','mode'])
def test_invalid_inputs_rejected_not_data_skips(mutation):
    x,a,_,v,t=batch();mode='native'
    if mutation=='zero_duration':t[0,0,1]=t[0,0,0]
    elif mutation=='negative_duration':t[0,0,1]=-10
    elif mutation=='nonfinite_time':t[0,0,0]=float('nan')
    elif mutation=='dtype':t=t.float()
    elif mutation=='nonfinite_feature':x[0,0,0]=float('inf')
    elif mutation=='logits':a[0,0,0]+=1
    elif mutation=='mask':v[1]=torch.tensor([1,0,1,0,0,0],dtype=torch.bool)
    elif mutation=='empty':v[0]=False
    else:mode='A_fallback'
    with pytest.raises(ValueError):IntervalTemporalHead()(x,a,v,t,mode=mode)

def test_center_only_preserves_each_interval_and_no_context():
    x,a,_,v,t=batch();m=nonneutral(IntervalTemporalHead()).eval()
    with torch.no_grad():
        out=m(x,a,v,t,mode='center_only')
        one=m(x[v][:,None],a[v][:,None],torch.ones(v.sum(),1,dtype=torch.bool),t[v][:,None])
    torch.testing.assert_close(out[v],one[:,0])

def rows(values):return [{'epoch':i,'val_macro_f1':v} for i,v in enumerate(values,1)]

def test_patience_strict_improvement_ties_first_plateau_and_no_extra_epoch():
    assert not plateau(rows([.7]+[.7]*9))['should_stop']
    assert plateau(rows([.7]+[.7]*10))['should_stop']
    state=plateau(rows([.7]+[.6]*9+[.70000001]))
    assert state['consecutive_no_improvement']==0 and state['best_epoch']==11
    with pytest.raises(ValueError):plateau(rows([.7]+[.7]*11))
    with pytest.raises(ValueError):plateau(rows([.7,float('nan')]))
