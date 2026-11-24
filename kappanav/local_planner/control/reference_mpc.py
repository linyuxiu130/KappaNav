"""The reference benchmark's evaluation controller (its BatchMPCController with the evaluation config N=30, ref_gap=3, T=0.1, v in [-0.5,0.5], w in [-0.5,0.5]): for each published plan it solves once in
the robot frame of that moment and plays the 30 controls back open loop, one per period, until the next plan."""
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import interp1d

N,REF_GAP,PERIOD_S,V_MAX,W_MAX=30,3,.1,.5,.5
REF_TRAJ_LENGTH_M,REF_DESIRED_V,MIN_DESIRED_V=2.,.5,.05


class ReferenceMPC:
    def __init__(self,codegen_dir,*,generate=True):
        import casadi as ca
        from acados_template import AcadosModel,AcadosOcp,AcadosOcpSolver
        self.period=PERIOD_S;self.max_speed=V_MAX
        folder=Path(codegen_dir).resolve()
        if generate:folder.mkdir(parents=True,exist_ok=False)
        x,y,theta,v,w=(ca.SX.sym(name) for name in ('x','y','theta','v','w'))
        states=ca.vertcat(x,y,theta);controls=ca.vertcat(v,w);rhs=ca.vertcat(v*ca.cos(theta),v*ca.sin(theta),w)
        model=AcadosModel();model.name='reference_mpc';model.x=states;model.u=controls
        model.xdot=ca.SX.sym('x_dot',3);model.f_expl_expr=rhs;model.f_impl_expr=model.xdot-rhs
        ocp=AcadosOcp();ocp.model=model
        ocp.solver_options.N_horizon=N;ocp.solver_options.tf=PERIOD_S*N
        ocp.cost.cost_type='LINEAR_LS';ocp.cost.cost_type_e='LINEAR_LS'
        Q=np.diag([10.,10.,0.]);R=np.diag([.05,.05])
        ocp.cost.W=np.block([[Q,np.zeros((3,2))],[np.zeros((2,3)),R]]);ocp.cost.W_e=Q
        ocp.cost.Vx=np.zeros((5,3));ocp.cost.Vx[:3,:3]=np.eye(3);ocp.cost.Vu=np.zeros((5,2));ocp.cost.Vu[3:,:]=np.eye(2)
        ocp.cost.Vx_e=np.eye(3);ocp.cost.yref=np.zeros(5);ocp.cost.yref_e=np.zeros(3)
        ocp.constraints.lbu=np.array([-V_MAX,-W_MAX]);ocp.constraints.ubu=np.array([V_MAX,W_MAX]);ocp.constraints.idxbu=np.array([0,1])
        ocp.constraints.x0=np.zeros(3)
        ocp.solver_options.nlp_solver_max_iter=100;ocp.solver_options.qp_solver_iter_max=100
        ocp.solver_options.qp_solver='FULL_CONDENSING_HPIPM';ocp.solver_options.hessian_approx='GAUSS_NEWTON'
        ocp.solver_options.integrator_type='ERK';ocp.solver_options.print_level=0;ocp.solver_options.nlp_solver_type='SQP'
        ocp.code_export_directory=str(folder)
        self.solver=AcadosOcpSolver(ocp,json_file=str(folder/'ocp.json'),generate=generate,build=generate,verbose=False)
        self.metadata=dict(controller='reference_mpc',horizon=N,ref_gap=REF_GAP,prediction_step_s=PERIOD_S,
            max_speed_mps=V_MAX,min_speed_mps=-V_MAX,max_angular_speed_rad_s=W_MAX,execution='open_loop_playback_per_plan')
        self.reset()

    def reset(self):
        self.controls=np.zeros((0,2))

    def publish(self,route,view):
        """Solve once for a newly published world route in the frame of `view`, the pose the plan was made from
        (as the reference evaluation does: its plans are robot-frame trajectories from the observation, with the
        origin prepended); the controls then play back from wherever the robot is."""
        c,s=np.cos(view[2]),np.sin(view[2]);local=(np.asarray(route,dtype=np.float64)-view[:2])@np.array([[c,-s],[s,c]])
        local=np.vstack((np.zeros(2),local))
        dense=_denser(local);length=float(np.linalg.norm(np.diff(local,axis=0),axis=1).sum())
        desired=min(max(MIN_DESIRED_V,min(V_MAX,REF_DESIRED_V*min(length/REF_TRAJ_LENGTH_M,1.))),
            _curvature_speed(_curvature(local)[:12].max()))
        count=N//REF_GAP+1;reference=_reference_points(dense,desired,count)
        reference=np.column_stack((reference,np.zeros(len(reference))))
        for i in range(N):
            self.solver.set(i,'x',np.zeros(3));self.solver.set(i,'u',np.array([1.,1.])*.001)
            low=min(int(np.floor(i/REF_GAP)),count-1);high=min(int(np.ceil(i/REF_GAP)),count-1);alpha=i/REF_GAP-low
            point=reference[low] if low==high else (1-alpha)*reference[low]+alpha*reference[high]
            self.solver.set(i,'yref',np.r_[point,0.,0.])
        self.solver.set(N,'yref',reference[-1]);self.solver.set(0,'lbx',np.zeros(3));self.solver.set(0,'ubx',np.zeros(3))
        began=time.perf_counter();self.status=self.solver.solve();self.solve_ms=1000*(time.perf_counter()-began)
        self.controls=np.array([self.solver.get(i,'u') for i in range(N)])

    def next_command(self,state,time_s):
        """The next control of the current plan's open-loop sequence (zero once it is used up)."""
        if len(self.controls):command,self.controls=self.controls[0].copy(),self.controls[1:]
        else:command=np.zeros(2)
        return command,dict(solve_ms=0.,sqp_iterations=0)


def _denser(route,ratio=50):
    index=np.arange(len(route));fine=np.linspace(0,len(route)-1,num=len(route)*ratio)
    return np.stack([interp1d(index,route[:,axis],kind='linear')(fine) for axis in range(2)],1)


def _curvature(route):
    dx,dy=np.gradient(route[:,0]),np.gradient(route[:,1]);dy[0]=0.
    ddx,ddy=np.gradient(dx),np.gradient(dy);denominator=np.maximum((dx**2+dy**2)**1.5,1e-6)
    return np.convolve(np.abs(dx*ddy-dy*ddx)/denominator,np.ones(3)/3,mode='same')


def _curvature_speed(curvature):
    return max(MIN_DESIRED_V,min(V_MAX,W_MAX/max(curvature,1e-6)))


def _reference_points(route,desired_v,count,lookahead_points=10):
    """The reference points ahead of the robot (the origin of the solve frame), as the reference controller picks them."""
    nearest=int(np.argmin(np.linalg.norm(route,axis=1)));start=min(nearest+lookahead_points,len(route)-1)
    spacing=desired_v*REF_GAP*PERIOD_S;arc=np.cumsum(np.linalg.norm(np.diff(route,axis=0),axis=1));points=[]
    for i in range(start,len(route)-1):
        if arc[i]-arc[start]>=spacing*len(points):
            points.append(route[i])
            if len(points)==count:break
    while len(points)<count:points.append(route[-1])
    return np.array(points)
