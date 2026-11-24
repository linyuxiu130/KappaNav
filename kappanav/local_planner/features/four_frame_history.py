"""Four recent observations;missing slots repeat the current acquisition."""
from collections import OrderedDict
from dataclasses import dataclass
import copy
import numpy as np
import torch
from kappanav.common.depth import PinholeIntrinsics
from kappanav.local_planner.features.obstacle_memory import VOXEL_M, volumes_observed_clear
from kappanav.local_planner.model.types import PolicyCondition
from kappanav.local_planner.features.staging import stage

HISTORY_CONTRACT = 'four_frame_2s_current_padding'
FRAMES, HISTORY_SECONDS = 4, 2.
READ_RADIUS_M = 12.


def sample_frames(timestamps, rng):
    """Current plus random recent acquisitions;pad missing slots with current."""
    times=np.asarray(timestamps,dtype=np.float64)
    candidates=np.flatnonzero((times[-1]-times>0)&(times[-1]-times<=HISTORY_SECONDS))
    chosen=np.sort(rng.choice(candidates,size=min(FRAMES-1,len(candidates)),replace=False))
    padding=FRAMES-1-len(chosen)
    return np.r_[np.full(padding,len(times)-1),chosen,len(times)-1].astype(np.int64),np.ones(FRAMES,bool)



@dataclass
class Observation:
    index: int
    time: float
    depth: np.ndarray
    pose: np.ndarray
    intrinsic: np.ndarray
    camera: np.ndarray


@torch.no_grad()
def project_frames(policy,frames):
    device=next(policy.parameters()).device;count=len(frames)
    depth,intrinsic,camera=stage([torch.as_tensor(np.stack([f.depth for f in frames])[:,None,None]),
        torch.as_tensor(np.stack([f.intrinsic for f in frames])[:,None]),
        torch.as_tensor(np.stack([f.camera for f in frames])[:,None])],device)
    observation=PolicyCondition(depth=depth,point_goal=torch.zeros(count,2,device=device),camera_intrinsics=intrinsic,
        camera_to_body=camera,observation_to_current=torch.eye(4,device=device)[None,None].expand(count,1,4,4),
        observation_age_s=torch.zeros(count,1,device=device),observation_valid=torch.ones(count,1,device=device,dtype=torch.bool),
        obstacle_memory=torch.zeros(count,64,64,device=device,dtype=torch.bool))
    projected=policy.depth_encoder.metric_projector(observation,with_field=False,dense_body=True)
    packed=torch.cat((projected.points,projected.pixel_indices[...,None],projected.depth[...,None],
        projected.obstacle_valid[...,None],projected.token_valid[...,None]),-1).cpu().numpy()[:,0]
    projections=[]
    for frame,values in zip(frames,packed,strict=True):
        values=values[values[:,6].astype(bool)];points=values[:,:3].astype(np.float64)
        projections.append(dict(world=points@frame.pose[:3,:3].T+frame.pose[:3,3],pixels=values[:,3].astype(np.int64),
            depth=values[:,4],kind=values[:,5].astype(bool)))
    return projections


class FourFrameHistory:
    """update buffers raw observations; write performs one sampled fusion event.

    Each write replaces the previous window with its exact immutable source plan.
    Repeated reads never write; newer free evidence refutes older selected frames.
    """
    def __init__(self,policy,*,capacity=32768,seed=0):
        self.policy=policy;self.device=next(policy.parameters()).device
        self.projector=policy.depth_encoder.metric_projector
        self.maximum=self.projector.max_depth_m
        self.resolution=2*self.projector.planning_horizon_m/63
        self.capacity=capacity;self.rng=np.random.default_rng(seed)
        self.observations=OrderedDict();self.cells=OrderedDict()
        self.visual={};self.maps={};self.projections={};self.writes=[]
        self.sequence=0;self.last_write=-1;self.clock=-float('inf')
        self.selection=None

    def update(self,depth,intrinsic,camera_to_body,body_to_world,timestamp):
        depth=np.asarray(depth,dtype=np.float32)
        PinholeIntrinsics.from_matrix(intrinsic,width=depth.shape[1],height=depth.shape[0])
        self.clock=float(timestamp);self.pose=np.asarray(body_to_world).copy()
        self.observations[self.sequence]=Observation(self.sequence,self.clock,depth.copy(),self.pose.copy(),np.asarray(intrinsic).copy(),np.asarray(camera_to_body).copy())
        self.sequence+=1
        expired=[i for i,frame in self.observations.items() if self.clock-frame.time>HISTORY_SECONDS]
        for i in expired:
            del self.observations[i];self.maps.pop(i,None);self.projections.pop(i,None)

    @torch.no_grad()
    def project(self,frame):
        if frame.index not in self.projections:
            self.projections[frame.index]=project_frames(self.policy,[frame])[0]
        return self.projections[frame.index]

    def _refute(self,records,frame):
        keys=[key for key,value in records.items() if value['time']<frame.time]
        if not keys:return
        low=np.stack([records[key]['low'] for key in keys]);high=np.stack([records[key]['high'] for key in keys])
        corners=np.array([[x,y,z] for x in (0,1) for y in (0,1) for z in (0,1)])
        clear=volumes_observed_clear(low[:,None]+corners*(high-low)[:,None],frame.depth,frame.intrinsic,frame.pose@frame.camera,self.maximum)
        for key in (key for key,flag in zip(keys,clear,strict=True) if flag):
            del records[key]

    def sample(self):
        frames=list(self.observations.values())
        ids,valid=sample_frames([frame.time for frame in frames],self.rng)
        return [frames[i].index for i in ids]

    @torch.no_grad()
    def plan_write(self,selected_indices):
        """Commit one explicit four-frame selection; replay uses the recorded selection unchanged."""
        if self.last_write==self.sequence-1:return []
        frames=list(self.observations.values());selected_indices=np.asarray(selected_indices,dtype=np.int64)
        lookup={frame.index:i for i,frame in enumerate(frames)}
        past=selected_indices[selected_indices!=self.sequence-1]
        padding=FRAMES-1-len(past)
        valid=np.ones(FRAMES,bool)
        self.cells.clear();self.visual.clear()
        ids=np.array([lookup[int(i)] for i in selected_indices])
        selected=[frames[i] for i in ids]
        self.selection=dict(indices=[frame.index for frame in selected],padding=np.arange(FRAMES)<padding,
            valid=valid,poses=np.stack([frame.pose for frame in selected]),ages=np.array([self.clock-frame.time for frame in selected],dtype=np.float32))
        window={}
        for frame in sorted(selected,key=lambda frame:frame.index):
            self._refute(window,frame)
            projection=self.project(frame);world=projection['world']
            if 'cell_sources' not in projection:
                keys=np.column_stack((np.floor(world[:,:2]/self.resolution+.5).astype(np.int64),projection['kind']))
                unique,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
                order=np.argsort(inverse,kind='stable');offsets=np.r_[0,np.cumsum(counts)]
                origin=(frame.pose@frame.camera)[:3,3]
                sources=[]
                for row,cell in enumerate(unique):
                    key=tuple(map(int,cell));indices=order[offsets[row]:offsets[row+1]];points=world[indices]
                    center=np.array([key[0]*self.resolution,key[1]*self.resolution,frame.pose[2,3]])
                    source=dict(frame=frame.index,pixels=projection['pixels'][indices],offsets=(points-center).astype(np.float32),
                        world=points,depth=projection['depth'][indices],origin=origin.copy(),time=frame.time,kind=key[2])
                    sources.append((key,source,points.min(0)-VOXEL_M/2,points.max(0)+VOXEL_M/2))
                projection['cell_sources']=sources
            for key,source,low,high in projection['cell_sources']:
                if key not in window:window[key]=dict(sources=[],low=low.copy(),high=high.copy(),time=frame.time)
                record=window[key];record['sources'].append(source);record['time']=frame.time
                record['low']=np.minimum(record['low'],low)
                record['high']=np.maximum(record['high'],high)
        changed=[]
        for key,record in window.items():
            record['write_time']=self.clock;cache_geometry(record)
            self.cells[key]=record;self.cells.move_to_end(key)
            changed.append(key)
        while len(self.cells)>self.capacity:
            key,_=self.cells.popitem(last=False);self.visual.pop(key,None)
        changed=[key for key in changed if key in self.cells]
        self.last_write=self.sequence-1
        self.writes.append(dict(acquisitions=self.sequence,indices=list(self.selection['indices'])))
        return changed

    @torch.no_grad()
    def write(self):
        from kappanav.local_planner.features.history_fusion import fuse_sources
        if self.last_write==self.sequence-1:return
        changed=self.plan_write(self.sample())
        if not changed:return
        plan=fusion_plan(self.cells,changed,self.pose,self.clock,self.selection)
        for index in np.unique(plan['frame_ids']):
            index=int(index)
            if index not in self.maps:
                frame=self.observations[index]
                self.maps[index]=self.policy.depth_encoder.image_features(torch.as_tensor(frame.depth[None,None],device=self.device))
        widths={index:self.observations[int(index)].depth.shape[1] for index in np.unique(plan['frame_ids'])}
        features=fuse_sources(self.policy,plan,self.maps,widths).detach().cpu()
        self.visual.update((key,value.clone()) for key,value in zip(changed,features,strict=True))

    def _read_keys(self):
        keys=list(self.cells)
        if not keys:return []
        world=np.stack([record['world_mean'] for record in self.cells.values()])
        points=(world-self.pose[:3,3])@self.pose[:3,:3]
        return sorted(key for key,valid in zip(keys,(np.abs(points[:,:2])<=READ_RADIUS_M).all(-1),strict=True) if valid)

    def read_plan(self):
        return fusion_plan(self.cells,self._read_keys(),self.pose,self.clock,self.selection)

    def packet(self):
        plan=memory_geometry(self.cells,self._read_keys(),self.pose,self.clock,self.selection)
        visual=torch.stack([self.visual[tuple(key)] for key in plan['keys']]) if len(plan['keys']) else torch.empty(0,self.policy.depth_encoder.spatial_projection.out_channels)
        return plan,visual.to(self.device)

    def state_dict(self):
        return dict(contract=HISTORY_CONTRACT,capacity=self.capacity,resolution=self.resolution,clock=self.clock,pose=self.pose.copy(),
            sequence=self.sequence,last_write=self.last_write,selection=copy.deepcopy(self.selection),
            cells=copy.deepcopy(self.cells),observations=copy.deepcopy(self.observations),
            visual={key:value.clone() for key,value in self.visual.items()},rng=copy.deepcopy(self.rng.bit_generator.state),writes=copy.deepcopy(self.writes))

    def load_state_dict(self,state):
        for name in ('clock','pose','sequence','last_write','selection','cells','observations','visual','writes'):
            setattr(self,name,copy.deepcopy(state[name]))
        for record in self.cells.values():cache_geometry(record)
        self.rng.bit_generator.state=copy.deepcopy(state['rng']);self.maps={};self.projections={}


def cache_geometry(record):
    """Cell geometry is immutable until the next write replaces its sources."""
    sources=record['sources'];record['pixel_count']=sum(len(source['pixels']) for source in sources)
    record['world_mean']=np.concatenate([source['world'] for source in sources]).mean(0)
    record['origin_mean']=sum(source['origin']*len(source['pixels']) for source in sources)/record['pixel_count']
    record['depth_mean']=np.concatenate([source['depth'] for source in sources]).mean()


def memory_geometry(cells,keys,pose,clock,selection):
    """Read cached cell geometry without expanding historical source pixels."""
    records=[cells[key] for key in keys]
    world=np.asarray([r['world_mean'] for r in records]).reshape(-1,3)
    origins=np.asarray([r['origin_mean'] for r in records]).reshape(-1,3)
    return dict(contract=HISTORY_CONTRACT,keys=np.asarray(keys,dtype=np.int64).reshape(-1,3),
        lengths=np.asarray([r['pixel_count'] for r in records],dtype=np.int64),
        points=((world-pose[:3,3])@pose[:3,:3]).astype(np.float32),
        origins=((origins-pose[:3,3])@pose[:3,:3]).astype(np.float32),
        depth=np.asarray([r['depth_mean'] for r in records],dtype=np.float32),
        times=np.asarray([r['time'] for r in records],dtype=np.float64),pose=pose.copy(),clock=clock,selection=copy.deepcopy(selection))


def fusion_plan(cells,keys,pose,clock,selection):
    """Compact, immutable replay of the exact source membership used at each write."""
    sources=[];age=[]
    for key in keys:
        record=cells[key];sources.extend(record['sources'])
        age.extend(record['write_time']-source['time'] for source in record['sources'])
    lengths=np.asarray([len(source['pixels']) for source in sources],dtype=np.int64)
    return dict(memory_geometry(cells,keys,pose,clock,selection),
        frame_ids=np.repeat(np.asarray([s['frame'] for s in sources],dtype=np.int64),lengths),
        pixels=np.concatenate([s['pixels'] for s in sources]).astype(np.int64,copy=False) if sources else np.empty(0,np.int64),
        offsets=np.concatenate([s['offsets'] for s in sources]).astype(np.float32,copy=False) if sources else np.empty((0,3),np.float32),
        depths=np.concatenate([s['depth'] for s in sources]).astype(np.float32,copy=False) if sources else np.empty(0,np.float32),
        kinds=np.repeat(np.asarray([s['kind'] for s in sources],dtype=np.float32),lengths),
        ages=np.repeat(np.asarray(age,dtype=np.float32),lengths))
