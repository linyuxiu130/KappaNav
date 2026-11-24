import numpy as np
import pytest
import torch
from kappanav.local_planner.features.history_fusion import _plan_inputs
from reference_implementations import source_features
from kappanav.local_planner.model.encoders.depth import DepthObservationEncoder


def test_derived_plan_inputs_are_bounded_and_eviction_preserves_pixel_groups():
    from kappanav.local_planner.features.history_fusion import source_groups, _PLAN_INPUTS, _PLAN_INPUT_LIMIT
    plan=dict(frame_ids=np.array([2,0,2,1,0]))
    expected=[(int(frame),indices.copy()) for frame,indices in source_groups(plan)]
    for _ in range(_PLAN_INPUT_LIMIT+44):source_groups(dict(frame_ids=np.array([0,1])))
    assert len(_PLAN_INPUTS)==_PLAN_INPUT_LIMIT and id(plan) not in _PLAN_INPUTS
    actual=source_groups(plan)
    for (frame,indices),(reference,positions) in zip(actual,expected,strict=True):
        assert frame==reference
        np.testing.assert_array_equal(indices,positions)
    assert set(plan)=={'frame_ids'}


def test_batched_source_sampling_matches_frame_sampling_and_gradients():
    torch.manual_seed(73)
    maps={i:torch.randn(1,4,3,6,requires_grad=True) for i in range(3)}
    reference={i:t.detach().clone().requires_grad_() for i,t in maps.items()}
    plan=dict(frame_ids=np.array([2,0,2,1,0,1,2,0]),pixels=np.array([0,16,47*89+88,88,32*89+64,16,0,47*89]))
    actual=source_features(DepthObservationEncoder,plan,maps,{i:89 for i in maps})
    expected=torch.stack([DepthObservationEncoder.sample_pixel_features(reference[int(i)],torch.tensor([[[int(p)]]]),89)[0,0,0] for i,p in zip(plan['frame_ids'],plan['pixels'],strict=True)])
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    probe=torch.randn_like(actual)
    (actual*probe).sum().backward();(expected*probe).sum().backward()
    for i in maps:torch.testing.assert_close(maps[i].grad,reference[i].grad,rtol=2e-5,atol=2e-6)
    cached=source_features(DepthObservationEncoder,plan,maps,{i:89 for i in maps})
    torch.testing.assert_close(actual,cached,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='calibrated CUDA interpolation')
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_cuda_source_sampling_is_bitexact_and_bounds_ragged_padding(dtype):
    rng=np.random.default_rng(73);torch.manual_seed(73)
    maps={i:torch.randn(1,32,8,14,device='cuda').to(dtype) for i in range(7)}
    plan=dict(frame_ids=rng.integers(0,7,10000),pixels=rng.integers(0,126*224,10000))
    actual=source_features(DepthObservationEncoder,plan,maps,{i:224 for i in maps})
    expected=torch.empty_like(actual)
    for i in maps:
        selected=np.flatnonzero(plan['frame_ids']==i)
        pixels=torch.as_tensor(plan['pixels'][selected],device='cuda')[None,None]
        expected[torch.as_tensor(selected,device='cuda')]=DepthObservationEncoder.sample_pixel_features(maps[i],pixels,224)[0,0].float()
    assert torch.equal(actual,expected)
    prepared,_=next(iter(_plan_inputs(plan)['_source_sample_inputs'].values()))
    assert sum(pixels.numel() for _,_,pixels,_ in prepared)<2*len(plan['pixels'])
    assert all(pixels.device.type==valid.device.type=='cpu' for _,_,pixels,valid in prepared)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_cross_state_sampling_preserves_pixels_shared_frames_and_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(73);rng=np.random.default_rng(73)
    maps={i:torch.randn(1,8,4,7,device=device,dtype=dtype,requires_grad=True) for i in range(3)}
    other={i:torch.randn(1,8,3,5,device=device,dtype=dtype,requires_grad=True) for i in range(2)}
    references=[{i:x.detach().clone().requires_grad_() for i,x in values.items()} for values in [maps,other]]
    sources=[maps,other,maps];reference_sources=[references[0],references[1],references[0]]
    widths=[112,80,112];plans=[]
    for row in range(3):
        plans.append(dict(frame_ids=rng.integers(0,len(sources[row]),37+row*19),
            pixels=rng.integers(0,(64 if row!=1 else 48)*widths[row],37+row*19)))
    from kappanav.local_planner.features.image_batch import packed_source_features_batch
    actual=packed_source_features_batch(DepthObservationEncoder,plans,sources,[{i:w for i in values} for values,w in zip(sources,widths,strict=True)])
    expected=[]
    for plan,values,width in zip(plans,reference_sources,widths,strict=True):
        result=torch.empty(len(plan['pixels']),8,device=device)
        for i,features in values.items():
            selected=np.flatnonzero(plan['frame_ids']==i)
            pixels=torch.as_tensor(plan['pixels'][selected],device=device)[None,None]
            result[torch.as_tensor(selected,device=device)]=DepthObservationEncoder.sample_pixel_features(features,pixels,width)[0,0].float()
        expected.append(result)
    for a,b in zip(actual,expected,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
    probes=[torch.randn_like(x) for x in actual]
    sum((x*p).sum() for x,p in zip(actual,probes,strict=True)).backward()
    sum((x*p).sum() for x,p in zip(expected,probes,strict=True)).backward()
    for values,reference in zip([maps,other],references,strict=True):
        for i in values:torch.testing.assert_close(values[i].grad,reference[i].grad,rtol=2e-5,atol=2e-6)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_training_ray_buckets_keep_calibrated_samples_and_gradients(device,dtype):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from kappanav.local_planner.features.history_fusion import source_sample_inputs
    from kappanav.local_planner.features.image_batch import packed_source_features_batch

    rng=np.random.default_rng(83);torch.manual_seed(83)
    maps={i:torch.randn(1,8,6,12,device=device,dtype=dtype,requires_grad=True) for i in range(3)}
    reference={i:x.detach().clone().requires_grad_() for i,x in maps.items()}
    plans=[]
    for counts in ((1050,600,360),(700,1200,400)):
        ids=np.repeat(np.arange(3),counts);rng.shuffle(ids)
        pixels=rng.integers(0,96*192,len(ids));pixels[:6]=[0,0,191,95*192,96*192-1,16]
        plans.append(dict(frame_ids=ids,pixels=pixels))
    widths=[{i:192 for i in maps} for _ in plans]
    # A plan can be used for both deployment and training: both derived forms
    # must coexist, including a real pixel zero and padded zero indices.
    shapes={i:tuple(v.shape) for i,v in maps.items()}
    default,_=source_sample_inputs(plans[0],shapes,widths[0])
    bucketed,_=source_sample_inputs(plans[0],shapes,widths[0],bucket=512)
    assert sorted(p.shape[-1] for _,_,p,_ in default)==[512,1024,2048]
    assert sorted(p.shape[-1] for _,_,p,_ in bucketed)==[512,1024,1536]
    cache=_plan_inputs(plans[0])
    assert len(cache['_source_sample_inputs'])==2

    old=packed_source_features_batch(DepthObservationEncoder,plans,[reference]*2,widths)
    new=packed_source_features_batch(DepthObservationEncoder,plans,[maps]*2,widths,bucket=512)
    for a,b in zip(old,new,strict=True):torch.testing.assert_close(a,b,rtol=0,atol=0)
    probes=[torch.randn_like(a) for a in old]
    sum((a*p).sum() for a,p in zip(old,probes,strict=True)).backward()
    sum((a*p).sum() for a,p in zip(new,probes,strict=True)).backward()
    # Regrouping changes accumulation order, including BF16 rounding across groups.
    for i in maps:
        expected=reference[i].grad.float();actual=maps[i].grad.float()
        relative=(actual-expected).norm()/expected.norm().clamp_min(1e-30)
        assert relative.item() < (0.01 if dtype==torch.bfloat16 else 1e-5)


@pytest.mark.parametrize('lengths',[[-1,4],[1,1]])
def test_source_fusion_rejects_invalid_lengths_before_gpu_staging(lengths):
    from types import SimpleNamespace
    from kappanav.local_planner.features.history_fusion import source_fusion_inputs
    plan=dict(lengths=np.asarray(lengths),frame_ids=np.array([0,0,0]))
    with pytest.raises(ValueError,match='Ray segment lengths'):
        source_fusion_inputs(SimpleNamespace(max_depth_m=5.),plan)
