import pytest
import torch

from experiments.hospital_temporal.model import TemporalVariant
from experiments.hospital_temporal_compare.model import TemporalHead
from experiments.hospital_mstcn_visual.model import VisualTemporalHead
from experiments.hospital_asformer_interval.model import IntervalTemporalHead
from experiments.hospital_asformer_anchor.model import AnchoredTemporalHead

torch.set_num_threads(2)


@pytest.mark.parametrize("name,expected", [
    ("plain_tcn",148935), ("short_tcn",148935), ("bigru",224263),
    ("mstcnpp",960092), ("asformer",1096732), ("mstcnpp_visual_r",1255004),
    ("asformer_interval",1096845), ("asformer_interval_self_reference",1096844),
])
def test_published_model_counts_and_finite_independent_synthetic_segments(name, expected):
    if name in {"plain_tcn", "short_tcn"}:
        model = TemporalVariant(name)
    elif name == "mstcnpp_visual_r":
        model = VisualTemporalHead()
    elif name == "asformer_interval":
        model = IntervalTemporalHead()
    elif name == "asformer_interval_self_reference":
        model = AnchoredTemporalHead()
    else:
        model = TemporalHead(name)
    assert sum(p.numel() for p in model.parameters()) == expected
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2,3,1543,generator=generator)
    valid = torch.tensor([[True,False,False],[True,True,True]])
    start = torch.arange(3).double()*3
    times = torch.stack([start,start+2],-1)[None].repeat(2,1,1)
    model.eval()
    with torch.no_grad():
        if name in {"asformer_interval","asformer_interval_self_reference"}:
            out = model(x,x[...,-7:],valid,times)
        else:
            out = model(x,x[...,-7:],valid)
    assert out.shape == (2,3,7)
    assert bool(torch.isfinite(out).all())
