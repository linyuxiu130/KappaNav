"""One geometry decoder over the current four-frame fused spatial features."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from kappanav.local_planner.model.precision import NEURAL_DTYPE
from kappanav.local_planner.features.obstacle_memory import VOXEL_M
from kappanav.local_planner.features.four_frame_history import READ_RADIUS_M
from kappanav.local_planner.model.encoders.configuration import metric_grid
from kappanav.local_planner.model.types import ConfigurationFeatures


class _Bilinear2x(torch.autograd.Function):
    """Separable transpose avoids generic scatter in the compiled backward."""
    @staticmethod
    def forward(ctx,value,size):
        return F.interpolate(value,size=size,mode='bilinear',align_corners=False)

    @staticmethod
    def backward(ctx,gradient):
        work=gradient if gradient.dtype==torch.float64 else gradient.float()
        even,odd=work[...,::2],work[...,1::2]
        left=torch.cat((even[...,:1],odd[...,:-1]),-1)
        right=torch.cat((even[...,1:],odd[...,-1:]),-1)
        work=(even+odd)*.75+(left+right)*.25
        even,odd=work[...,::2,:],work[...,1::2,:]
        above=torch.cat((even[...,:1,:],odd[...,:-1,:]),-2)
        below=torch.cat((even[...,1:,:],odd[...,-1:,:]),-2)
        return ((even+odd)*.75+(above+below)*.25).to(gradient.dtype),None


class FeatureGeometryDecoder(nn.Module):
    """Decode one body-aligned memory raster; no recent/older or executed-footprint branches."""
    def __init__(self,model_dim,horizon,radius,width=64,*,context_radius_m=READ_RADIUS_M):
        super().__init__();self.neural_dtype=NEURAL_DTYPE
        self.horizon,self.radius=horizon,radius
        self.context_radius_m=float(context_radius_m)
        resolution=2*horizon/63
        # Avoid an extra border cell from roundoff when context equals the horizon.
        padding=math.ceil(round((context_radius_m-horizon)/resolution,12))
        self.raster_size=32*math.ceil((64+2*padding)/32)
        self.raster_padding=(self.raster_size-64)//2
        axis=torch.arange(-self.raster_padding,64+self.raster_padding,dtype=torch.float32)
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        xy=torch.stack((x,y),-1)
        self.register_buffer('raster_local',((xy-31.5)*(2*horizon/63)).double(),persistent=False)
        self.register_buffer('raster_coordinates',(xy/31.5-1).permute(2,0,1)[None],persistent=False)
        token_dim=model_dim-8
        position,encoding=metric_grid(token_dim,16,horizon*self.raster_size/64)
        self.register_buffer('metric_position',position)
        self.register_buffer('metric_encoding',encoding)
        def block(cin,cout,stride=1):
            return nn.Sequential(nn.Conv2d(cin,cout,3,stride=stride,padding=1),nn.GroupNorm(8,cout),nn.SiLU())
        self.stem=nn.Sequential(nn.Conv2d(model_dim+5,width,1),nn.GroupNorm(8,width),nn.SiLU())
        self.down=nn.ModuleList(block(width,width,2) for _ in range(5))
        self.up=nn.ModuleList(block(2*width,width) for _ in range(5))
        self.readout=nn.Conv2d(width,2,1)
        self.token_projection=nn.Conv2d(width,token_dim,1)

    def project_memory(self,tokens):
        if tokens.shape[1]==0:
            return tokens[:,:,:1].expand(*tokens.shape[:2],self.stem[0].out_channels)
        with torch.autocast(device_type=tokens.device.type,dtype=self.neural_dtype):
            value=F.conv2d(tokens.transpose(1,2).unsqueeze(2),self.stem[0].weight[:,:-5])
            return value.squeeze(2).transpose(1,2).float()

    def forward(self,memory,pose,mass,*,memory_is_projected=False):
        batch,_,size,_=memory.shape
        context=(size-64)//2
        world=torch.einsum('hwj,bij->bhwi',self.raster_local.double(),pose[:,:2,:2].double())+pose[:,None,None,:2,3]
        phase=torch.remainder(world,VOXEL_M).float()/VOXEL_M
        calibration=torch.cat((torch.log1p(mass),self.raster_coordinates.expand(batch,-1,-1,-1),phase.permute(0,3,1,2)),1)
        with torch.autocast(device_type=memory.device.type,dtype=self.neural_dtype):
            if memory_is_projected:
                # The affine visual term commutes with normalized splatting; bias stays dense.
                affine=self.stem[0]
                value=(memory+F.conv2d(calibration,affine.weight[:,-5:],affine.bias).float()).to(self.neural_dtype)
                value=self.stem[2](self.stem[1](value))
            else:value=self.stem(torch.cat((memory,calibration),1))
            levels=[value]
            for layer in self.down:value=layer(value);levels.append(value)
            # The raster is a multiple of32, so all five skip sizes are exactly2x.
            for layer,skip in zip(self.up,reversed(levels[:-1]),strict=True):
                value=layer(torch.cat((_Bilinear2x.apply(value,skip.shape[-2:]),skip),1))
            all_logits=self.readout(value).float()
            tokens=self.token_projection(F.adaptive_avg_pool2d(value,(16,16))).flatten(2).transpose(1,2)
        logits=all_logits[:,:,context:context+64,context:context+64]
        clearance=logits[:,:1]*self.horizon
        resolution=2*self.horizon/63
        # Explicit edge copies also compile under deterministic algorithms on torch 2.7.
        horizontal=torch.cat((clearance[...,:1],clearance,clearance[...,-1:]),-1)
        vertical=torch.cat((clearance[:,:,:1],clearance,clearance[:,:,-1:]),-2)
        dx=(horizontal[...,2:]-horizontal[...,:-2])/(2*resolution)
        dy=(vertical[:,:,2:]-vertical[:,:,:-2])/(2*resolution)
        overlap=torch.sigmoid(-clearance/resolution)
        free=logits[:,1:2].sigmoid()
        support=overlap+(1-overlap)*free
        configuration=ConfigurationFeatures(tokens=tokens+self.metric_encoding.to(tokens.dtype),
            metric_position=self.metric_position.expand(batch,-1,-1),
            observed_fraction=F.adaptive_avg_pool2d(-torch.expm1(-mass),(16,16)).flatten(1))
        return torch.cat((clearance,dx,dy,support,overlap,free),1),logits,configuration


def geometry_reconstruction_loss(field,logits,target,*,distance_available=True):
    """Direct metric distance and observed-free supervision; support is derived.

    Distance targets refer to observed occupied sites, including in unknown
    regions. A positive distance never certifies free space without free evidence.
    Free evidence has its own Bernoulli output: its loss must not change an
    already correct distance merely to open or close a distance-dependent gate.
    """
    resolution=2*3.6/63
    distance=field[:,:1];truth=target[:,:1]
    spatial_gradient=sum(F.smooth_l1_loss(torch.diff(distance,dim=axis)/resolution,
        torch.diff(truth,dim=axis)/resolution,reduction='none').flatten(1).mean(-1) for axis in (-1,-2))/2
    metric=F.smooth_l1_loss(distance/resolution,truth/resolution,reduction='none').flatten(1).mean(-1)+spatial_gradient
    return ((metric*torch.as_tensor(distance_available,device=field.device)).mean()
        +F.binary_cross_entropy_with_logits(logits[:,1:2],target[:,5:6]))
