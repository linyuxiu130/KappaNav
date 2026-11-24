"""Cumulative reachable area certified by the same calibrated depth for both agents."""
from itertools import product
import numpy as np
from scipy.ndimage import label
from kappanav.local_planner.features.obstacle_memory import volumes_observed_clear
from kappanav.common.physical import MAXIMUM_TRAVERSABLE_HEIGHT_M, ROBOT_COLLISION_HEIGHT_M
from kappanav.evaluation.simulation.scene import body_pose


class ExplorationCoverage:
    def __init__(self,navigation,start,k,camera):
        self.navigation=navigation;self.k=k;self.camera=camera
        components,_=label(navigation.free)
        cell=navigation.grid.world_to_grid(start[:2])
        component=components[tuple(cell)]
        self.reachable=components==component
        self.known=np.zeros_like(self.reachable);self.initial=None
        self.times=[];self.areas=[]
        self.corners=np.asarray(list(product((0.,1.),repeat=3)))

    def update(self,depth,view,time_s,camera_to_world=None):
        """Add the cells this depth image certifies; the camera pose defaults to the level body at view."""
        grid=self.navigation.grid
        lo=np.maximum(grid.world_to_grid(np.asarray(view[:2])-5.5),0)
        hi=np.minimum(grid.world_to_grid(np.asarray(view[:2])+5.5)+1,grid.free.shape)
        pending=self.reachable[lo[0]:hi[0],lo[1]:hi[1]]&~self.known[lo[0]:hi[0],lo[1]:hi[1]]
        cells=np.argwhere(pending)+lo
        ground=0. if self.navigation.ground_z is None else self.navigation.ground_z[cells[:,0],cells[:,1]]
        lower=np.column_stack((grid.origin_xy+cells*grid.cell_size_m,np.broadcast_to(ground,(len(cells),))+MAXIMUM_TRAVERSABLE_HEIGHT_M))
        size=np.array([grid.cell_size_m,grid.cell_size_m,ROBOT_COLLISION_HEIGHT_M-MAXIMUM_TRAVERSABLE_HEIGHT_M])
        boxes=lower[:,None]+self.corners*size
        if camera_to_world is None:camera_to_world=body_pose(view,ground_z=self.navigation.ground_height(view[:2]))@self.camera
        clear=volumes_observed_clear(boxes,depth,self.k,camera_to_world,5.)
        seen=cells[clear];self.known[seen[:,0],seen[:,1]]=True
        if self.initial is None:self.initial=self.known.copy()
        self.times.append(float(time_s));self.areas.append(float((self.known&~self.initial).sum()*grid.cell_size_m**2))

    @property
    def initial_area_m2(self):return float(self.initial.sum()*self.navigation.spacing**2)


def coverage_curve(navigation,raw,k,camera):
    result=ExplorationCoverage(navigation,raw['acquired_views'][0],k,camera)
    for depth,view,timestamp in zip(raw['depths'],raw['acquired_views'],raw['timestamps'],strict=True):
        result.update(depth,view,timestamp)
    return result
