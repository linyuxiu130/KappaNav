"""Evaluate the benchmark in Isaac Sim: one process per scene (evaluation.isaac_scene), batch-one inference, and the
same plan selection, publication delay and controller as the Habitat entry."""
import argparse,json,os,sys,time
import multiprocessing as mp
from multiprocessing.connection import wait
from pathlib import Path
import numpy as np
from kappanav.evaluation.benchmark.protocol import DECISION_LATENCY_S, GOAL_BUDGET_S, decision_steps
from kappanav.evaluation.benchmark.results import write_summary
from kappanav.common.experiment_io import sha, write

CONTACT_FORCE_N=8.


def take_completed_scene(running):
    """Free a worker slot as soon as any scene finishes."""
    ready=wait([job[0].sentinel for job in running])
    index=next(i for i,job in enumerate(running) if job[0].sentinel in ready)
    return running.pop(index)


def level_body_pose(poses):
    """The gravity-aligned body frame at the actual root position, in which policy geometry is expressed."""
    out=np.repeat(np.eye(4,dtype=np.float32)[None],len(poses),axis=0)
    yaw=np.arctan2(poses[:,1,0],poses[:,0,0]);c,s=np.cos(yaw),np.sin(yaw)
    out[:,0,0],out[:,0,1],out[:,1,0],out[:,1,1]=c,-s,s,c
    out[:,:3,3]=poses[:,:3,3]
    return out


def scene_usd(scenes,row):
    return scenes/f"internscenes_{row['domain']}"/f"scenes_{row['domain']}"/row['asset_scene_id']/'start_result_navigation.usd'


def evaluate_scene(args,row,episodes,output):
    """One Isaac process per scene; the scene, weights and controller are reused across its episodes."""
    output.mkdir()
    with (output/'run.log').open('x') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES');os.environ.setdefault('OMNI_KIT_ALLOW_ROOT','1')
        from isaaclab.app import AppLauncher
        # Each scene owns one GPU; multi-GPU rendering duplicates scene resources across workers.
        app=AppLauncher(headless=True,enable_cameras=True,device=args.device,multi_gpu=False,kit_args=args.kit_args).app
        env=None
        try:
            import torch
            from pxr import UsdGeom
            from kappanav.evaluation.benchmark import isaac_scene
            from kappanav.local_planner.inference.bundle import load_bundle
            from kappanav.local_planner.control.reference_mpc import ReferenceMPC
            from kappanav.evaluation.simulation.navigation import NavigationGrid
            torch.set_num_threads(1)
            if any(e['goal_budget_s']!=GOAL_BUDGET_S for e in episodes):raise ValueError(f'Isaac episodes last {GOAL_BUDGET_S:g} s')
            usd=scene_usd(args.scenes,row);env=isaac_scene.IsaacScene(usd,args.robot,args.device)
            actor=load_bundle(args.checkpoint,args.policy_device);controller=ReferenceMPC(output/'mpc_codegen')
            decision_steps(args.decision_latency_s,env.step_dt);navigation=NavigationGrid(row['navigation'])
            # The camera mount in base_link, read from USD.
            xforms=UsdGeom.XformCache();prim=env.stage.GetPrimAtPath(env.camera._view.prim_paths[0])
            base=np.asarray(xforms.GetLocalToWorldTransform(prim.GetParent())).T;optical=np.asarray(xforms.GetLocalToWorldTransform(prim)).T
            base_rotation=base[:3,:3]/np.linalg.norm(base[:3,:3],axis=0);camera_rotation=optical[:3,:3]/np.linalg.norm(optical[:3,:3],axis=0)
            mount=np.eye(4);mount[:3,:3]=base_rotation.T@camera_rotation@np.diag([1.,-1.,-1.]);mount[:3,3]=base_rotation.T@(optical[:3,3]-base[:3,3])
            write(output/'inputs.json',dict(scene_id=row['scene_id'],checkpoint_sha256=sha(args.checkpoint),benchmark_sha256=sha(args.benchmark),
                scene_usd_sha256=sha(usd),robot_usd_sha256=sha(args.robot),code_sha256={name:sha(Path(m.__file__)) for name,m in
                (('evaluation.isaac',sys.modules[__name__]),('evaluation.isaac_scene',isaac_scene))},
                episode_ids=[e['episode_id'] for e in episodes],decision_latency_s=args.decision_latency_s,inference_batch_size=1,controller=controller.metadata))
            rows=[]
            for episode in episodes:
                env.reset(episode['start'],episode['goal']);controller.reset()
                folder=output/episode['episode_id'].rsplit('/',1)[-1];folder.mkdir()
                result=run_episode(app,env,actor,controller,navigation,mount,episode,args,folder)
                if args.exploration:result.update(run_exploration(app,env,actor,controller,navigation,mount,episode,args,folder))
                rows.append(result);write(folder/'result.json',result);print(json.dumps(result),flush=True)
            write(output/'rows.json',rows)
        finally:
            # Isaac's fast shutdown exits with status 0; never use it while unwinding an error.
            if env is not None:env.close()
            if sys.exc_info()[0] is None:app.close(wait_for_replicator=False)


class _Observer:
    """The robot pose and the policy-resolution depth image at the current step."""
    def __init__(self,env,mount):
        from kappanav.common.depth import PinholeIntrinsics
        self.env,self.mount=env,mount
        shape=env.camera.data.output['distance_to_image_plane'].shape
        self.source_k=PinholeIntrinsics.from_matrix(env.camera.data.intrinsic_matrices[0].cpu().numpy(),width=shape[2],height=shape[1])

    def pose(self):
        """(root pose, level body pose, camera in the level body frame)."""
        from scipy.spatial.transform import Rotation
        robot=self.env.robot;pose=np.eye(4,dtype=np.float32);q=robot.data.root_quat_w[0].cpu().numpy()
        pose[:3,:3]=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();pose[:3,3]=robot.data.root_pos_w[0].cpu().numpy()
        level=level_body_pose(pose[None])[0]
        return pose,level,(np.linalg.inv(level.astype(np.float64))@pose@self.mount).astype(np.float32)

    def depth(self):
        from kappanav.common.depth import preprocess_depth
        return preprocess_depth(self.env.camera.data.output['distance_to_image_plane'][0,...,0].cpu().numpy(),
            source_intrinsics=self.source_k,maximum_m=5.,height=126,width=224)


def _view(pose):
    return np.array([pose[0,3],pose[1,3],np.arctan2(pose[1,0],pose[0,0])])


def drive_policy(app,env,actor,controller,navigation,mount,episode,args,folder,*,goal_present=True,prefix='goal'):
    """Batch-one inference with delayed publication; the scene owns stepping, time-out and arrival. Writes
    PREFIX_candidates.npz, PREFIX_trace.npz and PREFIX_execution.json; returns (execution, raw candidates, records)."""
    import torch
    from kappanav.local_planner.inference.bundle import PlanCommitment
    from kappanav.evaluation.benchmark.metrics import mean_decision_latency_ms

    robot,contact,dt=env.robot,env.contact,env.step_dt;observer=_Observer(env,mount)
    history=actor.new_history();commitment=PlanCommitment()
    distance=0.;contacts=0;records=[];plans=[];traces=[];pending=None;step=0;next_observation=1
    interval=decision_steps(args.decision_latency_s,dt);status='running'

    def infer(observation):
        full,pose,extrinsic=observation;depth,k=observer.depth()
        depth=depth.astype(np.float16)
        goal=((np.asarray(episode['goal'])-pose[:2,3])@pose[:2,:2]).astype(np.float32) if goal_present else np.zeros(2,np.float32)
        history.update(depth,k,extrinsic,pose,step*dt)
        bank={key:value.cpu().numpy() for key,value in actor.step(history,goal,goal_present).items() if key!='selected_index'}
        records.append(dict(depth=depth,pose=pose,body_pose=full,camera=extrinsic,intrinsic=k,time_s=step*dt,
            candidate_paths=bank['paths'],candidate_values=bank['values'],candidate_scores=bank['scores'],candidate_clear=bank['clear'],
            selected=0,history_indices=np.asarray(history.selection['indices'])))
        return len(records)-1

    def publish(index,applied=True):
        record=records[index];pose=record['pose']
        selected,world=commitment.select(record['candidate_paths'],record['candidate_scores'],record['candidate_clear'],pose[:2,:2],pose[:2,3])
        record['selected']=selected
        plans.append(dict(observation_time_s=record['time_s'],time_s=step*dt,selected=selected,applied=applied,
            input_frame_ids=record['history_indices'].tolist(),decision_latency_ms=1000*(step*dt-record['time_s'])))
        if applied:controller.publish(world,_view(pose))

    with torch.no_grad(),torch.cuda.device(args.policy_device):
        while app.is_running() and status=='running':
            if step==next_observation:
                observation=observer.pose()
                if pending is None:
                    publish(infer(observation))
                    write(folder/'calibration.json',dict(intrinsic=observer.source_k.matrix().tolist(),camera_to_base_link=mount.tolist()))
                upcoming=infer(observation)
                if pending is not None:publish(pending)
                pending=upcoming;next_observation=step+interval
            full,pose,_=observer.pose();velocity=robot.data.root_lin_vel_w[0].cpu().numpy();view=_view(pose)
            force=contact.data.net_forces_w[0].cpu().numpy()
            if np.linalg.norm(force,axis=-1).max()>CONTACT_FORCE_N:contacts+=1
            command,_=controller.next_command(np.r_[view,velocity[:2]@np.array([np.cos(view[2]),np.sin(view[2])])],step*dt)
            distance+=float(np.linalg.norm(velocity[:2]))*dt
            traces.append((step*dt,full.copy(),velocity.copy(),command.copy(),force.copy()))
            status=env.step(command);step+=1
            if status=='running' and args.max_steps is not None and step>=args.max_steps:status='smoke_limit'
    if status=='running':raise RuntimeError('Isaac stopped before the episode finished')
    if pending is not None:publish(pending,False)
    raw={key:np.asarray([r[key] for r in records]) for key in ('candidate_paths','candidate_values','candidate_scores','candidate_clear')}
    raw.update(acquired_views=np.asarray([_view(r['pose']) for r in records]),selected_indices=np.asarray([r['selected'] for r in records]),
        depths=np.asarray([r['depth'] for r in records]),timestamps=np.asarray([r['time_s'] for r in records]))
    np.savez_compressed(folder/f'{prefix}_candidates.npz',**raw)
    if records:np.savez_compressed(folder/f'{prefix}_decisions.npz',**{key:np.stack([r[key] for r in records]) for key in records[0]})
    np.savez_compressed(folder/f'{prefix}_trace.npz',**{key:np.asarray([r[i] for r in traces]) for i,key in enumerate(('time_s','pose','velocity','command','contact_force'))})
    execution=dict(status=status,failure_reason=None,seconds=step*dt,path_m=distance,contacts=contacts,plans=plans,
        mean_decision_latency_ms=mean_decision_latency_ms(plans),controller=controller.metadata)
    write(folder/f'{prefix}_execution.json',execution)
    return execution,raw,records


def run_episode(app,env,actor,controller,navigation,mount,episode,args,folder):
    """The point-goal episode; env has been reset to the episode's start and goal."""
    from kappanav.evaluation.benchmark.metrics import goal_metrics
    from kappanav.evaluation.benchmark.trajectory_metrics import trajectory_metrics
    goal,raw,records=drive_policy(app,env,actor,controller,navigation,mount,episode,args,folder)
    generation=trajectory_metrics(navigation,raw,goal['plans'],args.policy_device) if records else None
    return dict(episode_id=episode['episode_id'],scene_id=episode['scene_id'],source='benchmark',
        goal=goal_metrics(goal['status']=='goal_reached',episode['spl_distance_m'],goal['path_m']),goal_trajectory=generation,
        goal_collision=goal['contacts']>0,goal_contacts=goal['contacts'],goal_status=goal['status'],goal_failure_reason=None,
        goal_decision_latency_ms=goal['mean_decision_latency_ms'])


def drive_exploration_expert(app,env,controller,navigation,mount,episode,args,folder):
    """The privileged frontier reference of evaluation.expert driven in this scene: every control step it observes,
    extends coverage, and replans to the farthest unobserved region once the current target is seen or reached."""
    from kappanav.evaluation.benchmark.coverage import ExplorationCoverage
    from kappanav.evaluation.benchmark.expert import TARGET_REACHED_M, exploration_route
    robot,contact,dt=env.robot,env.contact,env.step_dt;observer=_Observer(env,mount)
    coverage=None;target=None;plans=[];contacts=0;distance=0.;step=0;status='running';traces=[]
    while app.is_running() and status=='running':
        full,pose,_=observer.pose();view=_view(pose);depth,k=observer.depth()
        if coverage is None:coverage=ExplorationCoverage(navigation,view,k,None)
        coverage.update(depth,view,step*dt,(full@mount).astype(np.float32))
        if (target is None or coverage.known[tuple(navigation.grid.world_to_grid(target))]
                or np.linalg.norm(view[:2]-target)<TARGET_REACHED_M):
            route=exploration_route(navigation,coverage,view)
            if route is None:break
            controller.publish(route,view);target=route[-1]
            plans.append(dict(world_path=route.tolist(),observation_time_s=step*dt))
        velocity=robot.data.root_lin_vel_w[0].cpu().numpy();force=contact.data.net_forces_w[0].cpu().numpy()
        if np.linalg.norm(force,axis=-1).max()>CONTACT_FORCE_N:contacts+=1
        command,_=controller.next_command(np.r_[view,velocity[:2]@np.array([np.cos(view[2]),np.sin(view[2])])],step*dt)
        distance+=float(np.linalg.norm(velocity[:2]))*dt;traces.append((step*dt,full.copy(),velocity.copy(),command.copy(),force.copy()))
        status=env.step(command);step+=1
        if status=='running' and args.max_steps is not None and step>=args.max_steps:status='smoke_limit'
    failure=None
    if not plans:status,failure='planning_failure','no traversable cell near the start'
    np.savez_compressed(folder/'expert_trace.npz',**{key:np.asarray([r[i] for r in traces]) for i,key in enumerate(('time_s','pose','velocity','command','contact_force'))})
    execution=dict(status=status,failure_reason=failure,collision=contacts>0,contacts=contacts,plans=plans,controller=controller.metadata,
        times=coverage.times,areas=coverage.areas,initial_area_m2=coverage.initial_area_m2,path_m=distance,seconds=step*dt,
        reference='farthest_unobserved_geodesic_MPC',optimality='Privileged deterministic reference;not globally optimal exploration')
    write(folder/'expert_execution.json',execution)
    return execution


def run_exploration(app,env,actor,controller,navigation,mount,episode,args,folder):
    """Expert-relative exploration from the episode's start without a goal; the expert and the policy each run the
    episode's exploration budget in this scene, and coverage comes from each one's own depth images."""
    from kappanav.evaluation.benchmark.coverage import ExplorationCoverage
    from kappanav.evaluation.benchmark.metrics import exploration_metrics
    from kappanav.evaluation.benchmark.trajectory_metrics import trajectory_metrics
    budget=episode['exploration_budget_s']
    env.reset(episode['start'],None,budget);controller.reset()
    expert=drive_exploration_expert(app,env,controller,navigation,mount,episode,args,folder)
    env.reset(episode['start'],None,budget);controller.reset()
    exploring,raw,records=drive_policy(app,env,actor,controller,navigation,mount,episode,args,folder,goal_present=False,prefix='exploration')
    coverage=None
    for record in records:
        view=_view(record['pose'])
        if coverage is None:coverage=ExplorationCoverage(navigation,view,record['intrinsic'],None)
        coverage.update(record['depth'].astype(np.float32),view,record['time_s'],(record['body_pose']@mount).astype(np.float32))
    times,areas,initial=(coverage.times,coverage.areas,coverage.initial_area_m2) if coverage else ([0.],[0.],0.)
    np.savez_compressed(folder/'exploration_curve.npz',times=times,areas=areas,expert_times=expert['times'],expert_areas=expert['areas'])
    collision=exploring['contacts']>0
    scores=exploration_metrics(times,areas,expert['times'],expert['areas'],episode['reachable_area_m2'],collision,initial_area_m2=initial,
        expert_collision=expert['collision'],expert_failure=expert['status']=='planning_failure',failure=False)
    return dict(exploration=scores,exploration_trajectory=trajectory_metrics(navigation,raw,exploring['plans'],args.policy_device) if records else None,
        exploration_collision=collision,exploration_status=exploring['status'],expert_status=expert['status'],expert_collision=expert['collision'],
        exploration_failure_reason=None,expert_failure_reason=expert['failure_reason'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('data/model/policy.pt'))
    parser.add_argument('--benchmark',type=Path,default=Path('data/benchmark/tasks.json'))
    parser.add_argument('--scenes',type=Path,required=True,help='official scene USDs (internscenes_<domain>/scenes_<domain>/<scene>/)')
    parser.add_argument('--robot',type=Path,required=True,help='the robot USD (the benchmark dingo.usd)')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--decision-latency-s',type=float,default=DECISION_LATENCY_S)
    parser.add_argument('--workers',type=int,default=1,help='scene processes at once, spread over the visible GPUs')
    parser.add_argument('--kit-args',default='');parser.add_argument('--max-steps',type=int)
    parser.add_argument('--exploration',action='store_true',help='also run each episode without a goal against the exploration expert (E-SPL)')
    args=parser.parse_args()
    from kappanav.local_planner.control.mpc import MPC_PERIOD_S
    decision_steps(args.decision_latency_s,MPC_PERIOD_S)
    if args.output is None:args.output=Path('outputs/evaluation')/('isaac_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime()))
    for name in ('checkpoint','benchmark','scenes','robot','output'):setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    data=json.loads(args.benchmark.read_text());context=mp.get_context('spawn')
    jobs=[(row,[e for e in data['episodes'] if e['scene_id']==row['scene_id']]) for row in data['scenes']]
    jobs=[(row,episodes) for row,episodes in jobs if episodes];rows=[];running=[]
    import torch;gpus=torch.cuda.device_count()
    def finish(process,row,folder):
        process.join()
        if process.exitcode or not (folder/'rows.json').is_file():
            raise RuntimeError(f'Isaac scene {row["scene_id"]} did not finish (exit {process.exitcode}); see {folder}/run.log')
        rows.extend(json.loads((folder/'rows.json').read_text()))
    for slot,(row,episodes) in enumerate(jobs):
        if len(running)==args.workers:finish(*take_completed_scene(running))
        worker=argparse.Namespace(**vars(args),device=f'cuda:{slot%gpus}',policy_device=f'cuda:{slot%gpus}')
        folder=args.output/row['asset_scene_id']
        process=context.Process(target=evaluate_scene,args=(worker,row,episodes,folder));process.start()
        running.append((process,row,folder))
    for job in running:finish(*job)
    summary=write_summary(args.output,rows,data['episodes'],backend='isaac',inference_batch_size=1,
        checkpoint_sha256=sha(args.checkpoint),benchmark_sha256=sha(args.benchmark),decision_latency_s=args.decision_latency_s,exploration=args.exploration)
    print(json.dumps(summary['groups']['all']),flush=True)


if __name__=='__main__':main()
