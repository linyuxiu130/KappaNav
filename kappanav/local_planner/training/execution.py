"""Training-only kernels and compilation (configure_training_execution); they change how the step runs, not the
weights or the checkpoint format. Evaluation keeps the eager inference path."""
from functools import partial,wraps
import torch
import triton
from kappanav.local_planner.model.network import _geometry_bias as bias
from kappanav.local_planner.model.network import blocks


def register_training_bias_kernels():
    """Register the training geometry-bias kernels with tile sizes tuned per GPU generation."""
    weight_gradient_warps=2 if torch.cuda.get_device_capability()==(12,0) else 4
    @bias.geometric_bias.register_kernel('cuda')
    def forward(x,w1,b1,w2,b2):
        h,o=w1.shape[0],w2.shape[0];m=x.numel()//7
        output=bias._bias_output(x,o)
        bias._forward[(triton.cdiv(m,64),)](x.contiguous(),w1,b1,w2,b2,output,m,h,o,
            **bias._blocks(h,o),SCENE=x.shape[1]*x.shape[2] if x.ndim==4 else 0,num_warps=4,arch='sm80')
        return output

    @bias._geometric_bias_backward.register_kernel('cuda')
    def backward(x,w1,b1,w2,b2,gradient,need_x):
        h,o=w1.shape[0],w2.shape[0];m=x.numel()//7
        tile=dict(bias._blocks(h,o),BM=32)
        groups=16
        # Weight gradients split across hidden tiles; dx needs the full hidden reduction.
        if not need_x:
            tile['BH']=min(32,tile['BH']);groups=32
        hidden_tiles=triton.cdiv(h,tile['BH'])
        count=triton.cdiv(m,32*groups)
        dx=torch.empty(x.shape,device=x.device,dtype=x.dtype) if need_x else x.new_empty(0)
        partial=torch.empty((count,h*(8+o)+o),device=x.device,dtype=torch.float32)
        gradient=gradient.permute(0,3,1,2).contiguous() if x.ndim==4 else gradient.contiguous()
        bias._backward[(count*hidden_tiles,)](x.contiguous(),w1,b1,w2,gradient,dx if need_x else x,partial,m,h,o,
            **tile,NEED_X=need_x,GROUPS=groups,H_TILES=hidden_tiles,SCENE=x.shape[1]*x.shape[2] if x.ndim==4 else 0,num_warps=4 if need_x else weight_gradient_warps,arch='sm80')
        return (dx,*bias._weight_gradients(partial,w1,b1,w2,b2))


def _gradient_forward(forward):
    compiled=torch.compile(forward,fullgraph=True,dynamic=True,options={'emulate_precision_casts':True})
    @wraps(forward)
    def call(*args,**kwargs):
        # Monitoring keeps its original eager path and does not add compiler variants.
        return compiled(*args,**kwargs) if torch.is_grad_enabled() else forward(*args,**kwargs)
    return call


def configure_training_execution(actor):
    """Per-GPU training execution: tiled geometry-bias kernels (SM89/SM120), the packed cross-state pixel sampler for
    history fusion, fixed 128-query attention chunks, and gradient-only compiled transformer blocks. Returns its
    description, with the CNN chunk size and the sampler bucket the batches use."""
    capability=torch.cuda.get_device_capability(next(actor.policy.parameters()).device)
    device_bytes=torch.cuda.get_device_properties(next(actor.policy.parameters()).device).total_memory
    if capability==(12,0) and device_bytes>=64<<30:
        for block in actor.policy.trajectory_decoder.blocks:
            block.cross_attention.saved_bias_bytes=min(1<<30,device_bytes//(4*len(actor.policy.trajectory_decoder.blocks)*2))
    tiled=capability in ((8,9),(12,0))
    if tiled:register_training_bias_kernels()
    from kappanav.local_planner.features import history_fusion
    from kappanav.local_planner.features.image_batch import packed_source_features_batch
    bucket=512 if capability==(12,0) else None
    history_fusion.source_features_batch=partial(packed_source_features_batch,bucket=bucket)
    # Fixed 128-query chunks remove guards on unused workspace thresholds in G.
    blocks._BIAS_CHUNK_BYTES=0
    for block in (*actor.policy.trajectory_decoder.blocks,*actor.policy.trajectory_evaluator.blocks):
        block.forward=_gradient_forward(block.forward)
    return dict(transformer='compiled_dynamic_gradient_only',attention='query128_aligned16_role_bounded_saved_bias',
        geometry_bias='head_major_forward_warp4_backward_adjacent32_reduce32' if tiled else 'head_major_original_tiles',
        source_sampling='packed_cross_state_gather',source_sampling_bucket=bucket,cnn_batch_size=256 if capability==(12,0) else 64,
        generator_saved_bias_bytes=actor.policy.trajectory_decoder.blocks[0].cross_attention.saved_bias_bytes)
