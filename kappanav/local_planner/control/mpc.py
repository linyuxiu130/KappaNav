"""XY position tracking for a forward-only unicycle with actuator response.

Curve tangents initialize the first solve;they are not body-heading targets.
"""
from pathlib import Path
import time
import numpy as np
from numba import njit
from kappanav.common.physical import ROBOT, TURN_CENTRE_FORWARD_OFFSET_M
from kappanav.common.geometry import turn_rate_per_m

MAX_LINEAR_SPEED_MPS=.5
SPEED_RESPONSE_S=ROBOT.speed_response_s  # first-order forward-speed response, in the plant and in PathMPC's model
MPC_PERIOD_S=.1
MPC_HORIZON=15
MAX_ANGULAR_SPEED_RAD_S=.5
CROSS_TRACK_SCALE_M=.001
TURN_ENTER_RAD=np.deg2rad(60.)
TURN_EXIT_RAD=np.deg2rad(20.)
TURN_LOOKAHEAD_M=.4
HEADING_WEIGHT=100.  # 0.1 rad heading error has unit cost.
ANGULAR_INCREMENT_WEIGHT=1000.  # A 0.032 rad/s command change has unit cost.


class MPCSolveError(RuntimeError):
    """The solver failed or returned a command outside its physical bounds."""


@njit(cache=True,nogil=True)
def unicycle_step(state,command,dt,response_s):
    """Exact constant-command integration;state is world x,y,yaw,forward speed."""
    speed,omega=command
    phase=np.exp(1j*state[2])
    constant=speed*dt*np.sinc(omega*dt/(2*np.pi))*np.exp(1j*(state[2]+omega*dt/2))
    exponent=-1/response_s+1j*omega
    transient=(state[3]-speed)*phase*np.expm1(exponent*dt)/exponent
    delta=constant+transient
    return np.array([state[0]+delta.real,state[1]+delta.imag,state[2]+omega*dt,
        speed+(state[3]-speed)*np.exp(-dt/response_s)])


@njit(cache=True,nogil=True)
def axle_state(state):
    """base_link state (x,y,yaw,...) -> the same state at the turn centre (the Dingo's wheel axle midpoint; the
    Scout Mini's base_link), where the drive is exactly the unicycle above."""
    result=np.asarray(state,dtype=np.float64).copy()
    result[0]+=TURN_CENTRE_FORWARD_OFFSET_M*np.cos(state[2]);result[1]+=TURN_CENTRE_FORWARD_OFFSET_M*np.sin(state[2])
    return result


@njit(cache=True,nogil=True)
def body_step(state,command,dt,response_s):
    """The robot's base_link (the pose and the contact disc centre) under a constant command, as in Isaac: the
    unicycle moves the turn centre and base_link trails it, so an in-place turn moves base_link on a circle of
    radius TURN_CENTRE_FORWARD_OFFSET_M."""
    result=unicycle_step(axle_state(state),command,dt,response_s)
    result[0]-=TURN_CENTRE_FORWARD_OFFSET_M*np.cos(result[2]);result[1]-=TURN_CENTRE_FORWARD_OFFSET_M*np.sin(result[2])
    return result


@njit(cache=True,nogil=True)
def _nominal_rollout(initial,controls,period,response_s):
    states=np.empty((len(controls)+1,len(initial)),np.float64);states[0]=initial
    for i in range(len(controls)):
        states[i+1,:4]=unicycle_step(states[i,:4],controls[i],period,response_s)
        states[i+1,4:]=controls[i]
    return states


class PathMPC:
    """One acados solver,owned by one control thread;only its first command is applied."""
    def __init__(self,codegen_dir,*,response_s=SPEED_RESPONSE_S,max_speed_mps=MAX_LINEAR_SPEED_MPS,generate=True):
        import casadi as ca
        from acados_template import AcadosModel,AcadosOcp,AcadosOcpSolver
        self.response_s=float(response_s);self.max_speed=float(max_speed_mps)
        self.horizon=MPC_HORIZON;self.period=MPC_PERIOD_S;self.max_omega=MAX_ANGULAR_SPEED_RAD_S
        folder=Path(codegen_dir).resolve()
        if generate:folder.mkdir(parents=True,exist_ok=False)
        state=ca.SX.sym('state',6);control=ca.SX.sym('control',2)
        # Previous commands are bookkeeping states for a smooth input-increment cost.
        nodes,weights=np.polynomial.legendre.leggauss(4)
        at=(nodes+1)*self.period/2
        speed=control[0]+(state[3]-control[0])*ca.exp(-ca.DM(at)/self.response_s)
        heading=state[2]+control[1]*ca.DM(at)
        dx=self.period/2*ca.dot(ca.DM(weights),speed*ca.cos(heading))
        dy=self.period/2*ca.dot(ca.DM(weights),speed*ca.sin(heading))
        model=AcadosModel();model.name='curvenav_path_tracking';model.x=state;model.u=control
        model.disc_dyn_expr=ca.vertcat(state[0]+dx,state[1]+dy,state[2]+control[1]*self.period,
            control[0]+(state[3]-control[0])*ca.exp(-self.period/self.response_s),control)
        ocp=AcadosOcp();ocp.model=model
        ocp.solver_options.N_horizon=self.horizon;ocp.solver_options.tf=self.period*self.horizon
        ocp.cost.cost_type='LINEAR_LS';ocp.cost.cost_type_e='LINEAR_LS'
        ocp.cost.W=np.diag([CROSS_TRACK_SCALE_M**-2,10.,100.,ANGULAR_INCREMENT_WEIGHT,HEADING_WEIGHT])
        ocp.cost.W_e=np.diag([CROSS_TRACK_SCALE_M**-2,10.,HEADING_WEIGHT])
        ocp.cost.Vx=np.zeros((5,6));ocp.cost.Vx[:2,:2]=np.eye(2);ocp.cost.Vx[2:4,4:]=-np.eye(2);ocp.cost.Vx[4,2]=1.
        ocp.cost.Vu=np.zeros((5,2));ocp.cost.Vu[2:4]=np.eye(2)
        ocp.cost.Vx_e=np.zeros((3,6));ocp.cost.Vx_e[:2,:2]=np.eye(2);ocp.cost.Vx_e[2,2]=1.
        ocp.cost.yref=np.zeros(5);ocp.cost.yref_e=np.zeros(3)
        ocp.constraints.x0=np.zeros(6);ocp.constraints.idxbu=np.array([0,1])
        ocp.constraints.lbu=np.array([0.,-self.max_omega]);ocp.constraints.ubu=np.array([self.max_speed,self.max_omega])
        ocp.solver_options.qp_solver='FULL_CONDENSING_HPIPM';ocp.solver_options.hessian_approx='GAUSS_NEWTON'
        # Position-only tracking makes Q semidefinite;avoid Cholesky-only Riccati.
        ocp.solver_options.qp_solver_ric_alg=0;ocp.solver_options.qp_solver_cond_ric_alg=0
        for name in ('stat','eq','ineq','comp'):setattr(ocp.solver_options,'qp_solver_tol_'+name,1e-9)
        ocp.solver_options.integrator_type='DISCRETE'
        # One convex QP per feedback update.
        ocp.solver_options.nlp_solver_type='SQP_RTI';ocp.solver_options.qp_solver_iter_max=100
        ocp.solver_options.print_level=0;ocp.code_export_directory=str(folder)
        self.solver=AcadosOcpSolver(ocp,json_file=str(folder/'ocp.json'),generate=generate,build=generate,verbose=False)
        self.reset()
        self.metadata=dict(controller='acados_heading_forward_tracking',horizon=self.horizon,prediction_step_s=self.period,
            max_speed_mps=self.max_speed,max_angular_speed_rad_s=self.max_omega,linear_response_s=self.response_s,
            cross_track_scale_m=CROSS_TRACK_SCALE_M,contour_weight=CROSS_TRACK_SCALE_M**-2,lag_weight=10.,control_increment_weights=[100.,ANGULAR_INCREMENT_WEIGHT],
            heading_output_required=False,tracking_reference='local_path_contour_and_progress',solver='SQP_RTI',qp_tolerance=1e-9,
            heading_weight=HEADING_WEIGHT,minimum_command_speed_mps=0.,turn_enter_rad=TURN_ENTER_RAD,
            turn_exit_rad=TURN_EXIT_RAD,turn_lookahead_m=TURN_LOOKAHEAD_M,
            residual_scope='Nonlinear residuals are diagnostic;RTI does not converge the full NLP at each feedback update.',
            initial_guess='current_reference_kinematic_rollout')

    def reset(self):
        self.previous_command=None;self.route=None
        self.turning_in_place=False;self.turn_direction=0
        self.solver.reset(reset_qp_solver_mem=1)

    def publish(self,route,view):
        """Track a newly published world route from its start."""
        self.route=np.asarray(route,dtype=np.float64);self.arcs=np.r_[0.,np.linalg.norm(np.diff(self.route,axis=0),axis=-1).cumsum()]
        self.curvature=turn_rate_per_m(self.route);self.progress=0.;self.segment=0

    def next_command(self,state,time_s):
        reference=self.reference(state[:3]) if self.route is not None else None
        if reference is None:return np.zeros(2),dict(solve_ms=0.,sqp_iterations=0)
        return self.solve(state,reference,time_s)

    def reference(self,view):
        """The horizon's reference points along the route from the axle midpoint's progress: full speed up to what still
        stops at the route end, bounded by the turn rate on the horizon."""
        route,arcs,curvature,progress,segment=self.route,self.arcs,self.curvature,self.progress,self.segment
        view=axle_state(view)
        def projection(index):
            delta=route[index+1]-route[index]
            fraction=np.clip(np.dot(view[:2]-route[index],delta)/np.dot(delta,delta),0,1)
            arc=max(progress,float(arcs[index]+fraction*(arcs[index+1]-arcs[index])))
            point=np.array([np.interp(arc,arcs,route[:,axis]) for axis in range(2)])
            return arc,float(np.linalg.norm(point-view[:2]))

        if arcs[-1]==0:return None
        arc,distance=projection(segment)
        while segment+1<len(route)-1:
            following,following_distance=projection(segment+1)
            if following_distance>distance or (following_distance==distance and arc<arcs[segment+1]):break
            segment+=1;arc,distance=following,following_distance
        progress=arc
        if progress>=arcs[-1]:self.progress,self.segment=progress,segment;return None
        # Stop before the path ends, retaining the existing response and turn-rate limits.
        speed=min(self.max_speed,(arcs[-1]-progress)/(self.response_s+self.period))
        end=progress+speed*self.horizon*self.period
        stop=min(len(arcs),int(np.searchsorted(arcs,end,side='right'))+1)
        maximum=float(curvature[segment:stop].max())
        speed=min(speed,self.max_omega/max(maximum,1e-12))
        samples=np.minimum(progress+np.arange(self.horizon+1)*speed*self.period,arcs[-1])
        self.progress,self.segment=progress,segment
        return np.stack([np.interp(samples,arcs,route[:,axis]) for axis in range(2)],-1)

    def solve(self,state,reference,time_s):
        """state is base_link's (x,y,yaw,forward speed); the plan tracks the axle midpoint, the unicycle point."""
        state=axle_state(np.asarray(state,dtype=np.float64));reference=np.asarray(reference,dtype=np.float64)
        c,s=np.cos(state[2]),np.sin(state[2]);local=(reference-state[:2])@np.array([[c,-s],[s,c]])
        arc=np.r_[0.,np.linalg.norm(np.diff(local,axis=0),axis=1).cumsum()]
        if arc[-1]>1e-9:
            ahead=np.array([np.interp(min(TURN_LOOKAHEAD_M,arc[-1]),arc,local[:,axis]) for axis in range(2)])
            direction=ahead-local[0];error=float(np.arctan2(direction[1],direction[0]))
        else:error=0.
        if self.turning_in_place:
            if abs(error)<=TURN_EXIT_RAD:self.turning_in_place=False
        elif abs(error)>=TURN_ENTER_RAD:self.turning_in_place=True
        previous=np.array([state[3],0.]) if self.previous_command is None else self.previous_command
        initial=np.r_[0.,0.,0.,state[3],previous]
        delta=np.diff(local,axis=0);nonzero=np.flatnonzero(np.linalg.norm(delta,axis=-1)>0)
        heading=np.interp(np.arange(self.horizon),nonzero,np.unwrap(np.arctan2(delta[nonzero,1],delta[nonzero,0]))) if len(nonzero) else np.zeros(self.horizon)
        speed_guess=np.minimum(np.linalg.norm(delta,axis=-1)/self.period,self.max_speed)*np.maximum(np.cos(heading),0.)
        guess=np.column_stack((speed_guess,np.full(self.horizon,
            np.clip(heading[0]/(self.horizon*self.period),-self.max_omega,self.max_omega))))
        # The feed-forward primal guess is fresh;old duals belong to a different reference.
        self.solver.reset(reset_qp_solver_mem=1)
        nominal=_nominal_rollout(initial,guess,self.period,self.response_s)
        segments=np.diff(local,axis=0);length2=(segments*segments).sum(-1)
        offset=nominal[:,None,:2]-local[:-1]
        fraction=((offset*segments).sum(-1)/np.maximum(length2,1e-20)).clip(0,1)
        closest=local[:-1]+fraction[...,None]*segments
        distance=((nominal[:,None,:2]-closest)**2).sum(-1)
        distance[:,length2==0]=np.inf
        selected=distance.argmin(-1)
        tangents=np.divide(segments,np.sqrt(length2[:,None]),out=np.tile([1.,0.],(len(segments),1)),where=length2[:,None]>0)[selected]
        heading_targets=np.unwrap(np.arctan2(tangents[:,1],tangents[:,0]))
        heading_targets-=2*np.pi*np.rint(heading_targets[0]/(2*np.pi))
        normals=np.column_stack((-tangents[:,1],tangents[:,0]))
        basis=np.stack((normals,tangents),1)
        # Lateral error uses the nominal path footpoint;time lag may relax during turning.
        targets=np.column_stack(((normals*closest[np.arange(len(nominal)),selected]).sum(-1),(tangents*local).sum(-1)))
        self.solver.set_flat('x',nominal.reshape(-1));self.solver.set_flat('u',guess.reshape(-1))
        projections=np.zeros((self.horizon,5,6));projections[:,:2,:2]=basis[:-1]
        projections[:,2:4,4:]=-np.eye(2);projections[:,4,2]=1.
        references=np.column_stack((targets[:-1],np.zeros((self.horizon,2)),heading_targets[:-1]))
        for i in range(self.horizon):
            self.solver.cost_set(i,'Vx',projections[i],api='new')
            self.solver.set(i,'yref',references[i])
        self.solver.set(self.horizon,'yref',np.r_[targets[-1],heading_targets[-1]])
        terminal=np.zeros((3,6));terminal[:2,:2]=basis[-1];terminal[2,2]=1.
        self.solver.cost_set(self.horizon,'Vx',terminal,api='new')
        self.solver.set(0,'lbx',initial);self.solver.set(0,'ubx',initial)
        began=time.perf_counter();status=self.solver.solve();elapsed=time.perf_counter()-began
        if status!=0:raise MPCSolveError(f'Acados tracking solve failed,status={status},residuals={self.solver.get_residuals()}')
        controls=self.solver.get_flat('u').reshape(self.horizon,2)
        command=controls[0].copy()
        if not np.isfinite(command).all() or command[0]<-1e-5 or command[0]>self.max_speed+1e-5 or abs(command[1])>self.max_omega+1e-5:
            raise MPCSolveError(f'MPC returned a nonfinite or out-of-bound command:{command},residuals={self.solver.get_residuals()}')
        self.previous_command=command.copy()
        residuals=self.solver.get_residuals()
        turning=bool(self.turning_in_place and abs(command[0])<.02 and abs(command[1])>.1)
        self.turn_direction=int(np.sign(command[1])) if turning else 0
        return command,dict(solve_ms=1000*elapsed,sqp_iterations=int(self.solver.get_stats('sqp_iter')),
            stationarity_residual=float(residuals[0]),maximum_feasibility_residual=float(np.max(residuals[1:])),
            control_bound_residual=float(max(np.maximum(abs(controls)-[self.max_speed,self.max_omega],0).max(),
                np.maximum(-controls[:,0],0).max())),
            turning_in_place=turning,turn_direction=self.turn_direction,turning_required=self.turning_in_place,
            heading_error_rad=error)
