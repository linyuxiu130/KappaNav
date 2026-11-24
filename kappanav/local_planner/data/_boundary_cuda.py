"""FP64 CUDA nearest-boundary selection with GPU block compaction."""
import torch
import triton
import triton.language as tl

@triton.jit
def _pack(V,IDS,COUNTS,WIDTH,NB,BS:tl.constexpr,NR:tl.constexpr):
    batch=tl.program_id(0);blocks=tl.arange(0,NR);cols=tl.arange(0,BS)
    segments=blocks[:,None]*BS+cols[None,:]
    flags=tl.sum(tl.load(V+batch*WIDTH+segments,(blocks[:,None]<NB)&(segments<WIDTH),0).to(tl.int32),1)>0
    position=tl.cumsum(flags.to(tl.int32))-1
    tl.store(IDS+batch*NB+position,blocks,flags)
    tl.store(COUNTS+batch,tl.sum(flags.to(tl.int32),0))

@triton.jit
def _compact_nearest(P,A,D,L,V,IDS,COUNTS,OUT,POINTS,WIDTH,NB,AB:tl.constexpr,AS:tl.constexpr,AC:tl.constexpr,DB:tl.constexpr,DS:tl.constexpr,DC:tl.constexpr,LB:tl.constexpr,LS:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0);batch=row//POINTS
    px=tl.load(P+row*2).to(tl.float64);py=tl.load(P+row*2+1).to(tl.float64)
    best=tl.full((),float('inf'),tl.float64);chosen=tl.full((),2147483647,tl.int32)
    count=tl.load(COUNTS+batch)
    for slot in range(count):
        block=tl.load(IDS+batch*NB+slot);j=block*BS+tl.arange(0,BS)
        valid=tl.load(V+batch*WIDTH+j,j<WIDTH,0)
        ax=tl.load(A+batch*AB+j*AS,j<WIDTH,0);ay=tl.load(A+batch*AB+j*AS+AC,j<WIDTH,0)
        dx=tl.load(D+batch*DB+j*DS,j<WIDTH,0);dy=tl.load(D+batch*DB+j*DS+DC,j<WIDTH,0)
        length=tl.load(L+batch*LB+j*LS,j<WIDTH,1)
        x=px-ax;y=py-ay;t=tl.minimum(tl.maximum((x*dx+y*dy)/length,0.),1.)
        rx=x-t*dx;ry=y-t*dy;distance=tl.where(valid,rx*rx+ry*ry,float('inf'))
        local=tl.min(distance,0);index=tl.min(tl.where(distance==local,j,2147483647),0)
        take=(local<best)|((local==best)&(index<chosen))
        chosen=tl.where(take,index,chosen);best=tl.minimum(best,local)
    tl.store(OUT+row,chosen.to(tl.int64))


def nearest_valid_boundary(world,a,delta,length2,valid):
    batch,points=world.shape[:2];width=a.shape[-2];blocks=triton.cdiv(width,128)
    ids=torch.empty((batch,blocks),device=world.device,dtype=torch.int32)
    counts=torch.empty(batch,device=world.device,dtype=torch.int32)
    output=torch.empty((batch,points),device=world.device,dtype=torch.long)
    _pack[(batch,)](valid.contiguous(),ids,counts,width,blocks,BS=128,NR=triton.next_power_of_2(blocks),num_warps=4)
    _compact_nearest[(batch*points,)](world.contiguous(),a,delta,length2,valid.contiguous(),ids,counts,output,points,width,blocks,AB=a.stride(0) if a.ndim==3 else 0,AS=a.stride(-2),AC=a.stride(-1),DB=delta.stride(0) if delta.ndim==3 else 0,DS=delta.stride(-2),DC=delta.stride(-1),LB=length2.stride(0) if length2.ndim==2 else 0,LS=length2.stride(-1),BS=128,num_warps=4,enable_fp_fusion=False)
    return output
