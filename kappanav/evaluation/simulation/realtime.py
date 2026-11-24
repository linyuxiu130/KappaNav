"""Closed-loop motion on simulation time: the robot plant, contact, arrival, and latency-correct plan handover."""
import numpy as np
from kappanav.local_planner.control.mpc import body_step, MPCSolveError, SPEED_RESPONSE_S
from kappanav.common.physical import ROBOT, TURN_CENTRE_FORWARD_OFFSET_M

GOAL_TOLERANCE_M=.5
# The reference benchmark's arrival rule: at each ARRIVAL_TICK_S environment step,
# the first time the robot is within GOAL_TOLERANCE_M with L1 world speed |vx|+|vy| below ARRIVAL_SPEED_MPS a timer
# latches, and the episode succeeds ARRIVAL_HOLD_S (40 steps) later even if the robot has moved on; within a step
# the termination is checked before the latch.
ARRIVAL_SPEED_MPS,ARRIVAL_HOLD_S,ARRIVAL_TICK_S=.25,4.,.1


def goal_entry(points,goal):
    if np.linalg.norm(points[0]-goal)<=GOAL_TOLERANCE_M:return 0.
    delta=np.diff(points,axis=0);offset=points[:-1]-goal
    a=(delta*delta).sum(-1);b=2*(offset*delta).sum(-1);c=(offset*offset).sum(-1)-GOAL_TOLERANCE_M**2
    discriminant=b*b-4*a*c
    root=(-b-np.sqrt(np.maximum(discriminant,0)))/np.maximum(2*a,1e-20)
    valid=(a>0)&(discriminant>=0)&(root>=0)&(root<=1)
    distances=np.r_[0.,np.sqrt(a).cumsum()]
    return float(np.min(distances[:-1][valid]+root[valid]*np.sqrt(a[valid]))) if valid.any() else None


class RealtimeMotion:
    """Continuous world pose;planner publication and MPC adoption never reset motion."""
    def __init__(self,view,clearance,duration_s,goal=None,*,controller,initial_velocity=None,clock=None,collision_ends=True,arrival_hold_s=None):
        self.view=np.asarray(view,dtype=np.float64).copy();self.clearance=clearance
        self.goal=None if goal is None else np.asarray(goal,dtype=np.float64)
        self.controller=controller;self.response_s=SPEED_RESPONSE_S
        self.controller.reset()
        velocity=np.zeros(2) if initial_velocity is None else np.asarray(initial_velocity,dtype=np.float64)
        direction=np.array([np.cos(self.view[2]),np.sin(self.view[2])]);self.speed=float(velocity@direction)
        self.initial_velocity_residual=float(np.linalg.norm(velocity-self.speed*direction))
        self.command=np.array([self.speed,0.]);self.yaw_rate=0.;self.clock=ManualClock() if clock is None else clock;self.origin=self.clock();self.duration=float(duration_s);self.time=0.;self.path=0.
        self.status='running';self.failure_reason=None;self.trace=[];self.control_trace=[]
        # collision_ends=False: contact stops the body at the obstacle (speed zero) and the episode goes on, as a rigid
        # obstacle does in a physics simulator; motion that leaves the contact is free again.
        self.collision_ends=collision_ends;self.contacts=0
        # arrival_hold_s: the latched arrival (see ARRIVAL_SPEED_MPS), not success on first entry.
        self.arrival_hold_s=arrival_hold_s;self.latched_s=None;self.tick=ARRIVAL_TICK_S
        if self.goal is not None and np.linalg.norm(self.view[:2]-self.goal)<=GOAL_TOLERANCE_M:self.status='initial_goal_reached'

    @property
    def velocity(self):
        """base_link's world velocity: the axle's forward velocity minus the turn about the axle."""
        c,s=np.cos(self.view[2]),np.sin(self.view[2])
        return self.speed*np.array([c,s])-TURN_CENTRE_FORWARD_OFFSET_M*self.yaw_rate*np.array([-s,c])

    def _state(self):return np.r_[self.view,self.speed]

    def _advance(self,now):
        if self.status!='running':return
        target=min(now-self.origin,self.duration)
        while self.time<target and self.status=='running':
            dt=min(1/120,target-self.time);start=self._state();command=self.command.copy();command[1]*=yaw_response(start[3])
            endpoint=body_step(start,command,dt,self.response_s)
            acceleration_bound=abs(command[0]-start[3])/self.response_s+max(abs(command[0]),abs(start[3]))*abs(command[1])+TURN_CENTRE_FORWARD_OFFSET_M*command[1]**2
            def clearance_at(elapsed):
                pose=body_step(start,command,elapsed,self.response_s)[:3]
                padding=acceleration_bound*elapsed*elapsed/8
                return float(self.clearance(np.stack((start[:3],pose)),padding))
            # A rigid body overlapping an obstacle can still move out of it: motion may not go deeper than where it
            # starts. From any free start the floor is 0, the plain contact test.
            inside=float(self.clearance(np.stack((start[:3],start[:3])),0.))
            floor=0. if inside>=0 else inside-acceleration_bound*dt*dt/8
            attempted_dt=dt;attempted_margin=clearance_at(dt)
            entry=goal_entry(np.stack((start[:2],endpoint[:2])),self.goal) if self.goal is not None and self.arrival_hold_s is None else None
            goal_dt=None
            if entry is not None:
                goal_dt=dt*entry/max(float(np.linalg.norm(endpoint[:2]-start[:2])),1e-20)
                if np.linalg.norm(endpoint[:2]-self.goal)<=GOAL_TOLERANCE_M:
                    low,high=0.,dt
                    for _ in range(28):
                        middle=(low+high)/2;point=body_step(start,command,middle,self.response_s)[:2]
                        if np.linalg.norm(point-self.goal)<=GOAL_TOLERANCE_M:high=middle
                        else:low=middle
                    goal_dt=high
                elif np.linalg.norm(body_step(start,command,goal_dt,self.response_s)[:2]-self.goal)>GOAL_TOLERANCE_M:goal_dt=None
            moved=dt
            if goal_dt is not None and clearance_at(goal_dt)>=floor:
                dt=moved=goal_dt;self.status='goal_reached'
            elif attempted_margin<floor:
                low,high=0.,dt
                for _ in range(32):
                    middle=(low+high)/2
                    if clearance_at(middle)<floor:high=middle
                    else:low=middle
                moved=low
                if self.collision_ends:dt=low;self.status='collision'
            actual=body_step(start,command,moved,self.response_s);self.view[:]=actual[:3];self.speed=float(actual[3]);self.yaw_rate=float(command[1])
            if moved<dt:
                # Held at the contact for the rest of the step: no translation; the turn as _contact_pivot allows.
                self.speed=0.;self.yaw_rate=0.;self.contacts+=1
                turned=_contact_pivot(self.view,command[1]*(dt-moved),self.clearance,floor)
                if turned is not None:actual=np.r_[turned,0.];self.view[:]=turned;self.yaw_rate=float(command[1])
            self.path+=float(np.linalg.norm(actual[:2]-start[:2]));self.time+=dt
            while self.arrival_hold_s is not None and self.status=='running' and self.time>=self.tick-1e-9:
                if self.latched_s is not None and self.tick-self.latched_s>=self.arrival_hold_s-1e-9:self.status='goal_reached'
                elif self.latched_s is None and np.linalg.norm(self.view[:2]-self.goal)<GOAL_TOLERANCE_M and np.abs(self.velocity).sum()<ARRIVAL_SPEED_MPS:
                    self.latched_s=self.tick
                self.tick+=ARRIVAL_TICK_S
            c,s=np.cos(self.view[2]),np.sin(self.view[2])
            acceleration=(command[0]-self.speed)/self.response_s*np.array([c,s])+self.speed*command[1]*np.array([-s,c])
            self.trace.append(dict(time_s=self.time,view=self.view.tolist(),velocity_mps=self.velocity.tolist(),
                linear_speed_mps=self.speed,yaw_rate_rad_s=self.yaw_rate,command=command.tolist(),acceleration_mps2=acceleration.tolist(),
                path_m=self.path,status=self.status,sweep_margin_m=attempted_margin if moved==attempted_dt else clearance_at(moved),attempted_sweep_margin_m=attempted_margin,
                numerical_sweep_padding_m=acceleration_bound*dt*dt/8))
        if self.status=='running' and self.time>=self.duration:
            self.status='budget_exhausted'
            if self.trace:self.trace[-1]['status']=self.status

    def snapshot(self):
        self._advance(self.clock())
        return dict(time_s=self.time,view=self.view.copy(),velocity_mps=self.velocity,linear_speed_mps=self.speed,
            yaw_rate_rad_s=self.yaw_rate,status=self.status,path_m=self.path,failure_reason=self.failure_reason)

    def publish(self,world_path,observation_time,observation_view=None):
        """Hand a plan to the controller; observation_view is the pose the plan was made from (default: now)."""
        world_path=np.asarray(world_path,dtype=np.float64)
        self._advance(self.clock())
        if self.status!='running':return dict(applied=False,time_s=self.time,view=self.view.tolist())
        self.controller.publish(world_path[np.r_[True,np.linalg.norm(np.diff(world_path,axis=0),axis=-1)>0]],
            self.view.copy() if observation_view is None else np.asarray(observation_view,dtype=np.float64))
        return dict(applied=True,time_s=self.time,view=self.view.tolist(),velocity_mps=self.velocity.tolist(),observation_age_s=self.time-observation_time)

    def control_step(self):
        self._advance(self.clock())
        if self.status!='running':return
        state=self._state();observed=self.time
        command,statistics=self.controller.next_command(state,observed)
        self._advance(self.clock())
        if self.status!='running':return
        self.command=command.copy()
        self.control_trace.append(dict(time_s=self.time,state_time_s=observed,view=self.view.tolist(),command=command.tolist(),
            actuation_latency_ms=1000*(self.time-observed),**statistics))

    def control_steps(self,count):
        """Up to `count` controller periods of simulation time, each a control step and then the clock advance;
        stops when the episode ends. A solver failure ends a running episode as controller_failure with its
        reason; a terminal physical status it arrives after is kept."""
        try:
            for _ in range(count):
                if self.status!='running':return
                self.control_step();self.clock.time=min(self.clock.time+self.controller.period,self.duration)
        except MPCSolveError as error:
            if self.status=='running':self.status='controller_failure'
            self.failure_reason=str(error)


# The yaw rate the robot reaches for a commanded one, by forward speed (medians measured in Isaac physics; the
# Dingo's casters and tyres scrub turning in place and reach about half the commanded rate).
YAW_RESPONSE_SPEED_MPS=ROBOT.yaw_response_speed_mps
YAW_RESPONSE_GAIN=ROBOT.yaw_response_gain


def yaw_response(speed):
    return float(np.interp(abs(speed),YAW_RESPONSE_SPEED_MPS,YAW_RESPONSE_GAIN))


# Contact: the body cannot translate into or along an obstacle, but a turn about the turn centre still happens when the
# obstacle only has to push base_link back by at most this much along its outward direction (the wall's reaction
# while the wheels slip).
CONTACT_PIVOT_PUSH_M=.02


def _contact_pivot(view,angle,clearance,floor):
    """The pose after turning `angle` about the turn centre at a contact, with base_link pushed back along the obstacle's
    outward direction just far enough to clear it; None when that push exceeds CONTACT_PIVOT_PUSH_M."""
    if angle==0:return None
    d=TURN_CENTRE_FORWARD_OFFSET_M;x,y,yaw=view
    axle=np.array([x,y])+d*np.array([np.cos(yaw),np.sin(yaw)]);heading=yaw+angle
    point=axle-d*np.array([np.cos(heading),np.sin(heading)])
    value=lambda q,yaw=heading:float(clearance(np.stack(((q[0],q[1],yaw),)*2),0.))
    if value(point)>=floor:return np.array([point[0],point[1],heading])
    # The outward direction at the contact: of 32 directions, the one that gains the most clearance 1 cm out.
    directions=np.stack((np.cos(np.arange(32)*np.pi/16),np.sin(np.arange(32)*np.pi/16)),-1)
    normal=directions[int(np.argmax([value(np.array([x,y])+.01*u,yaw) for u in directions]))]
    if value(point+CONTACT_PIVOT_PUSH_M*normal)<floor:return None
    low,high=0.,CONTACT_PIVOT_PUSH_M
    for _ in range(24):
        middle=(low+high)/2
        if value(point+middle*normal)>=floor:high=middle
        else:low=middle
    moved=point+high*normal
    return np.array([moved[0],moved[1],heading])


class ManualClock:
    """Simulation time advanced explicitly; compute time never enters it."""
    def __init__(self):self.time=0.
    def __call__(self):return self.time
