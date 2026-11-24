"""Calibrated sensor input and exact procedural-scene collision geometry."""
from kappanav.common.experiment_io import sha
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np

from kappanav.common.depth import PinholeIntrinsics, preprocess_depth
from kappanav.evaluation.simulation import renderer
from kappanav.evaluation.simulation.renderer import set_pose
from kappanav.common.physical import DINGO, ROBOT, ROBOT_BASE_HEIGHT_ABOVE_GROUND_M, ROBOT_FOOTPRINT_RADIUS_M

create_simulator=renderer.create_simulator   # the simulator factory, re-exported for scene users

RUNTIME_ASSETS_PATH = None
RUNTIME_ASSETS_SHA256 = None
ROBOT_RADIUS = ROBOT_FOOTPRINT_RADIUS_M
# The selected robot's camera and body contract (configs/runtime.json is the Dingo's).
DEFAULT_RUNTIME = Path('configs') / ('runtime.json' if ROBOT is DINGO else f'runtime_{ROBOT.name}.json')


def verify_runtime(runtime_assets):
    data=json.loads(Path(runtime_assets).read_text())
    body=(data['body_radius_m'],data['body_base_height_m'])
    if not np.allclose(body,(ROBOT_FOOTPRINT_RADIUS_M,ROBOT_BASE_HEIGHT_ABOVE_GROUND_M),rtol=0,atol=1e-6):
        raise ValueError(f'{runtime_assets} describes a body {body}, not the selected robot {ROBOT.name}')
    return data


def configure(runtime_assets):
    global RUNTIME_ASSETS_PATH, RUNTIME_ASSETS_SHA256
    RUNTIME_ASSETS_PATH = Path(runtime_assets).resolve()
    RUNTIME_ASSETS_SHA256 = sha(RUNTIME_ASSETS_PATH)
    verify_runtime(RUNTIME_ASSETS_PATH)
    sensor_assets.cache_clear();rendered_intrinsics.cache_clear()
    sensor_assets()

MAX_DEPTH_M=5.0


@lru_cache(maxsize=1)
def rendered_intrinsics():
    """The rendered depth image's pinhole (principal point at the image centre)."""
    c=sensor_assets()[0]['camera']
    return PinholeIntrinsics(width=int(c['image_width']),height=int(c['image_height']),fx=float(c['focal_x_px']),
        fy=float(c['focal_y_px']),cx=c['image_width']/2,cy=c['image_height']/2)


@lru_cache(maxsize=1)
def sensor_assets():
    data=json.loads(RUNTIME_ASSETS_PATH.read_text())
    return data,np.asarray(data['intrinsics'],np.float32),np.asarray(data['camera_to_body'],np.float32)


def clearance(paths,boxes,radius=ROBOT_RADIUS):
    """Exact swept-disk segment-to-box clearance."""
    paths=np.asarray(paths,dtype=np.float64)
    if paths.ndim==2:paths=paths[None]
    boxes=np.asarray(boxes,dtype=np.float64)[None]
    a,b=paths[None,:,:-1,None,:],paths[None,:,1:,None,:]
    lo,hi=boxes[:,None,None,:,:2],boxes[:,None,None,:,2:]
    delta=b-a
    near=np.divide(lo-a,delta,out=np.full_like(lo-a,-np.inf),where=delta!=0)
    far=np.divide(hi-a,delta,out=np.full_like(hi-a,np.inf),where=delta!=0)
    enter=np.maximum(np.minimum(near,far).max(-1),0)
    leave=np.minimum(np.maximum(near,far).min(-1),1)
    intersects=(enter<=leave)&~(((delta==0)&((a<lo)|(a>hi))).any(-1))
    endpoint=np.minimum(np.linalg.norm(np.maximum(np.maximum(lo-a,a-hi),0),axis=-1),
        np.linalg.norm(np.maximum(np.maximum(lo-b,b-hi),0),axis=-1))
    corners=np.stack((lo,np.stack((lo[...,0],hi[...,1]),-1),hi,
        np.stack((hi[...,0],lo[...,1]),-1)),-2)
    fraction=(((corners-a[...,None,:])*delta[...,None,:]).sum(-1)/
        np.maximum((delta**2).sum(-1)[...,None],1e-20)).clip(0,1)
    corner_distance=np.linalg.norm(corners-(a[...,None,:]+fraction[...,None]*delta[...,None,:]),axis=-1).min(-1)
    return np.where(intersects,0,np.minimum(endpoint,corner_distance)).min((2,3))[0]-radius


def body_pose(view,ground_z=0.):
    x,y,angle=np.asarray(view,dtype=np.float64)
    pose=np.eye(4);c,s=math.cos(angle),math.sin(angle)
    pose[:2,:2]=[[c,-s],[s,c]]
    pose[:3,3]=[x,y,ground_z+ROBOT_BASE_HEIGHT_ABOVE_GROUND_M]
    return pose.astype(np.float32)


def raw_observation(sim,view,*,ground_z=0.,rgb=False):
    """The rendered image at body pose view at full resolution, for planners that take it as it is: depth in metres
    (camera z; inf where nothing was hit, as a real sensor reports no return) and, with `rgb`, colour (H x W x 3 uint8)."""
    set_pose(sim,np.asarray([view[0],ground_z,-view[1]],dtype=np.float32),-float(view[2]))
    observations=sim.get_sensor_observations()
    depth=np.asarray(observations['depth'],np.float32).copy();depth[~(depth>0)]=np.inf
    return (np.ascontiguousarray(np.asarray(observations['color'])[...,:3]) if rgb else None),depth


def observation(sim,view,k,camera=None,*,ground_z=0.):
    """Depth at body pose view; camera (renderer settings) moves the sensor first, else the simulator's own pose."""
    set_pose(sim,np.asarray([view[0],ground_z,-view[1]],dtype=np.float32),-float(view[2]))
    if camera is not None:renderer.place_camera(sim,camera)
    raw=np.asarray(sim.get_sensor_observations()['depth'])
    depth,measured_k=preprocess_depth(raw,source_intrinsics=rendered_intrinsics(),
        maximum_m=MAX_DEPTH_M,height=126,width=224)
    np.testing.assert_allclose(measured_k,k,rtol=0,atol=1e-5)
    return depth.astype(np.float16)
