"""BF16 geometry MLP fusion without per-pair hidden tensors."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit(do_not_specialize=['SCENE'])
def _forward(X,W1,B1,W2,B2,Y,M,H:tl.constexpr,O:tl.constexpr,
             BM:tl.constexpr,BH:tl.constexpr,BO:tl.constexpr,SCENE=0):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    ii=tl.arange(0,16);hh=tl.arange(0,BH);oo=tl.arange(0,BO)
    x=tl.load(X+rows[:,None]*7+ii[None,:],(rows[:,None]<M)&(ii[None,:]<7),0)
    w1=tl.load(W1+hh[None,:]*7+ii[:,None],(hh[None,:]<H)&(ii[:,None]<7),0).to(tl.bfloat16)
    b1=tl.load(B1+hh,hh<H,0).to(tl.bfloat16).to(tl.float32)
    z=tl.dot(x,w1)+b1[None,:]
    z=tl.where(rows[:,None]<M,z,0).to(tl.bfloat16).to(tl.float32)
    hidden=(z/(1+libdevice.exp(-z))).to(tl.bfloat16)
    w2=tl.load(W2+oo[None,:]*H+hh[:,None],(oo[None,:]<O)&(hh[:,None]<H),0).to(tl.bfloat16)
    b2=tl.load(B2+oo,oo<O,0).to(tl.bfloat16).to(tl.float32)
    y=tl.dot(hidden,w2)+b2[None,:]
    offsets=rows[:,None]*O+oo[None,:]
    if SCENE:
        offsets=(rows[:,None]//SCENE)*(SCENE*O)+(rows[:,None]%SCENE)+oo[None,:]*SCENE
    tl.store(Y+offsets,y,(rows[:,None]<M)&(oo[None,:]<O))

@triton.jit
def _backward(X,W1,B1,W2,DY,DX,PARTIAL,M,H:tl.constexpr,O:tl.constexpr,
              BM:tl.constexpr,BH:tl.constexpr,BO:tl.constexpr,NEED_X:tl.constexpr,GROUPS:tl.constexpr,SCENE:tl.constexpr=0,H_TILES:tl.constexpr=1):
    # Neighboring hidden tiles reuse the same x/dy rows.
    block=tl.program_id(0)//H_TILES
    hidden_tile=tl.program_id(0)%H_TILES
    ii=tl.arange(0,16);hh=hidden_tile*BH+tl.arange(0,BH);oo=tl.arange(0,BO)
    w1=tl.load(W1+hh[None,:]*7+ii[:,None],(hh[None,:]<H)&(ii[:,None]<7),0).to(tl.bfloat16)
    b1=tl.load(B1+hh,hh<H,0).to(tl.bfloat16).to(tl.float32)
    w2t=tl.load(W2+oo[:,None]*H+hh[None,:],(oo[:,None]<O)&(hh[None,:]<H),0).to(tl.bfloat16)
    gw1=tl.full((BH,16),0,tl.float32);gb1=tl.full((BH,),0,tl.float32)
    gw2=tl.full((BO,BH),0,tl.float32);gb2=tl.full((BO,),0,tl.float32)
    # Accumulate row tiles on chip before writing gradient partials.
    for group in range(GROUPS):
        rows=(block*GROUPS+group)*BM+tl.arange(0,BM)
        x=tl.load(X+rows[:,None]*7+ii[None,:],(rows[:,None]<M)&(ii[None,:]<7),0)
        z=tl.dot(x,w1)+b1[None,:]
        z=tl.where(rows[:,None]<M,z,0).to(tl.bfloat16).to(tl.float32)
        sigmoid=1/(1+libdevice.exp(-z));hidden=(z*sigmoid).to(tl.bfloat16)
        offsets=rows[:,None]*O+oo[None,:]
        if SCENE:
            offsets=(rows[:,None]//SCENE)*(SCENE*O)+(rows[:,None]%SCENE)+oo[None,:]*SCENE
        dy=tl.load(DY+offsets,(rows[:,None]<M)&(oo[None,:]<O),0)
        dh=tl.dot(dy,w2t).to(tl.bfloat16).to(tl.float32)
        dz=(dh*sigmoid*(1+z*(1-sigmoid))).to(tl.bfloat16)
        if NEED_X:
            dx=tl.dot(dz,tl.trans(w1)).to(tl.bfloat16)
            tl.store(DX+rows[:,None]*7+ii[None,:],dx,(rows[:,None]<M)&(ii[None,:]<7))
        gw1=tl.dot(tl.trans(dz),x,gw1)
        gb1+=tl.sum(dz.to(tl.float32),axis=0)
        gw2=tl.dot(tl.trans(dy),hidden,gw2)
        gb2+=tl.sum(dy.to(tl.float32),axis=0)
    packed=H*(8+O)+O
    tl.store(PARTIAL+block*packed+hh[:,None]*7+ii[None,:],gw1,(hh[:,None]<H)&(ii[None,:]<7))
    tl.store(PARTIAL+block*packed+7*H+hh,gb1,hh<H)
    tl.store(PARTIAL+block*packed+8*H+oo[:,None]*H+hh[None,:],gw2,(oo[:,None]<O)&(hh[None,:]<H))
    tl.store(PARTIAL+block*packed+(8+O)*H+oo,gb2,(oo<O)&(hidden_tile==0))


def _blocks(h,o):
    return dict(BM=64,BH=max(16,triton.next_power_of_2(h)),BO=max(16,triton.next_power_of_2(o)))


def _bias_output(x, heads):
    # SDPA reads [batch, heads, queries, keys] without a full bias transpose.
    if x.ndim==4:
        batch,queries,keys,_=x.shape
        return x.new_empty((batch,heads,queries,keys)).permute(0,2,3,1)
    return x.new_empty((*x.shape[:-1],heads))


@triton.jit
def _pack_weight_gradients(SUM,GW1,GB1,GW2,GB2,H:tl.constexpr,O:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    total=H*(8+O)+O
    # Retain neural BF16 rounding before writing parameter-dtype gradients.
    value=tl.load(SUM+index,index<total,0).to(tl.bfloat16).to(tl.float32)
    tl.store(GW1+index,value,index<7*H)
    tl.store(GB1+index-7*H,value,(index>=7*H)&(index<8*H))
    tl.store(GW2+index-8*H,value,(index>=8*H)&(index<(8+O)*H))
    tl.store(GB2+index-(8+O)*H,value,(index>=(8+O)*H)&(index<total))


def _weight_gradients(partial,w1,b1,w2,b2):
    h,o=w1.shape[0],w2.shape[0]
    result=tuple(torch.empty_like(weight) for weight in (w1,b1,w2,b2))
    _pack_weight_gradients[(triton.cdiv(h*(8+o)+o,256),)](partial.sum(0),*result,h,o,BLOCK=256)
    return result


# Opaque custom ops: torch.compile keeps the attention block in one graph around the kernels.
@torch.library.custom_op('curvenav::geometric_bias',mutates_args=())
def geometric_bias(x:torch.Tensor,w1:torch.Tensor,b1:torch.Tensor,w2:torch.Tensor,b2:torch.Tensor)->torch.Tensor:
    """Per-pair bias MLP 7->H->O with SiLU, without materializing per-pair hidden tensors."""
    h,o=w1.shape[0],w2.shape[0];m=x.numel()//7
    output=_bias_output(x,o)
    # MMA2 lowering supports Triton3.3/SM120; final assembly targets the current GPU.
    _forward[(triton.cdiv(m,64),)](x.contiguous(),w1,b1,w2,b2,output,m,h,o,**_blocks(h,o),SCENE=x.shape[1]*x.shape[2] if x.ndim==4 else 0,num_warps=4,arch='sm80')
    return output


@geometric_bias.register_fake
def _(x,w1,b1,w2,b2):
    return _bias_output(x,w2.shape[0])


@torch.library.custom_op('curvenav::geometric_bias_backward',mutates_args=())
def _geometric_bias_backward(x:torch.Tensor,w1:torch.Tensor,b1:torch.Tensor,w2:torch.Tensor,b2:torch.Tensor,
                             gradient:torch.Tensor,need_x:bool)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    h,o=w1.shape[0],w2.shape[0];m=x.numel()//7;blocks=triton.cdiv(m,256)
    dx=torch.empty(x.shape,device=x.device,dtype=x.dtype) if need_x else x.new_empty(0)
    partial=torch.empty((blocks,h*(8+o)+o),device=x.device,dtype=torch.float32)
    gradient=gradient.permute(0,3,1,2).contiguous() if x.ndim==4 else gradient.contiguous()
    _backward[(blocks,)](x.contiguous(),w1,b1,w2,gradient,dx if need_x else x,partial,m,h,o,
        **_blocks(h,o),NEED_X=need_x,GROUPS=4,SCENE=x.shape[1]*x.shape[2] if x.ndim==4 else 0,num_warps=4,arch='sm80')
    return (dx,*_weight_gradients(partial,w1,b1,w2,b2))


@_geometric_bias_backward.register_fake
def _(x,w1,b1,w2,b2,gradient,need_x):
    return (torch.empty_like(x) if need_x else x.new_empty(0),torch.empty_like(w1),torch.empty_like(b1),
            torch.empty_like(w2),torch.empty_like(b2))


def _setup_context(ctx,inputs,output):
    ctx.save_for_backward(*inputs)


def _backward_rule(ctx,gradient):
    dx,*weights=_geometric_bias_backward(*ctx.saved_tensors,gradient,ctx.needs_input_grad[0])
    return (dx if ctx.needs_input_grad[0] else None,*weights)


geometric_bias.register_autograd(_backward_rule,setup_context=_setup_context)
