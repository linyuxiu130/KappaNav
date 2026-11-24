"""The evaluation: SR/SPL on the benchmark task set; optional expert-relative exploration."""
from pathlib import Path
import argparse,json,os,signal,sys,time
import multiprocessing as mp
from multiprocessing.connection import wait
import numpy as np
import torch
from kappanav.common.experiment_io import sha, write
from kappanav.evaluation.benchmark.coverage import coverage_curve
from kappanav.evaluation.benchmark.expert import run_exploration_expert
from kappanav.evaluation.benchmark.metrics import goal_metrics, exploration_metrics, mean_decision_latency_ms
from kappanav.evaluation.benchmark.protocol import DECISION_LATENCY_S, decision_steps
from kappanav.evaluation.benchmark.trajectory_metrics import trajectory_metrics
from kappanav.evaluation.simulation import scene
from kappanav.local_planner.control.reference_mpc import ReferenceMPC
from kappanav.local_planner.control.mpc import PathMPC
from kappanav.evaluation.simulation.realtime import RealtimeMotion
from kappanav.local_planner.inference.bundle import PlanCommitment
from kappanav.global_planner.route import Subgoals, plan_routes



# Workers pull episodes from one queue, longest expected first; summaries keep benchmark task order.
EXECUTION_ORDER='shared_queue_longest_expected_first'

def save_execution(folder,name,result):
    trace=result.pop('motion_trace')
    np.savez_compressed(folder/(name+'_trace.npz'),**{key:np.asarray([row[key] for row in trace]) for key in (trace[0] if trace else ())})
    (folder/(name+'_execution.json')).write_text(json.dumps(result)+'\n')


DECISION_KEYS=('candidate_paths','candidate_values','candidate_scores','candidate_clear','selected_indices','acquired_views','timestamps','depths')


class CurveNavPlanner:
    """CurveNav's decisions: the four-frame history, the model server's candidate bank and PlanCommitment."""
    rgb=False

    def __init__(self,actor):self.actor=actor

    def start(self,sim):
        self.sim=sim;self.history=self.actor.new_history();self.commitment=PlanCommitment()
        self.decisions={key:[] for key in DECISION_KEYS}

    def request(self,view,ground_z,pose,timestamp,goal,present):
        """Observe and request a plan; the returned callable completes the decision as (world path, plan record)."""
        _,k,camera=scene.sensor_assets()
        depth=scene.observation(self.sim,view,k,ground_z=ground_z)
        self.history.update(depth,k,camera,pose,timestamp)
        if hasattr(self.actor,'submit'):receive=self.actor.submit(self.history,goal,present)
        else:
            bank=self.actor.step(self.history,goal,present);receive=lambda:bank
        view=np.asarray(view).copy();indices=self.history.selection['indices']  # the next request replaces the selection
        def complete():
            bank=receive()
            paths=bank['paths'].cpu().numpy()
            c,s=np.cos(view[2]),np.sin(view[2])
            selected,world=self.commitment.select(paths,bank['scores'].float().cpu().numpy(),bank['clear'].cpu().numpy(),
                np.array([[c,-s],[s,c]]),np.asarray(view[:2]))
            for key,value in zip(DECISION_KEYS,(paths,bank['values'].cpu().numpy(),bank['scores'].float().cpu().numpy(),
                    bank['clear'].cpu().numpy(),selected,view,float(timestamp),depth)):
                self.decisions[key].append(value)
            return world,dict(selected=selected,input_frame_ids=list(indices))
        return complete

    def finish(self):return {key:np.asarray(value) for key,value in self.decisions.items()}


class BaselinePlanner:
    """A baseline policy server's decisions (evaluation.baselines): full-resolution colour and depth in, a trajectory
    out in the planar frame the method plans in (at its camera or at the robot base, heading the body's), tracked by
    the same controller."""
    def __init__(self,client):self.client=client;self.rgb=client.rgb

    def start(self,sim):
        intrinsics=scene.rendered_intrinsics()
        self.sim,self.k=sim,np.array([[intrinsics.fx,0.,intrinsics.cx],[0.,intrinsics.fy,intrinsics.cy],[0.,0.,1.]])
        # The frame origin in the level body frame: the camera's position, or the base itself.
        self.origin=np.asarray(scene.sensor_assets()[2][:2,3],np.float64) if self.client.origin=='camera' else np.zeros(2)
        self.client.reset(self.k);self.decisions=dict(acquired_views=[],timestamps=[],trajectories=[])

    def request(self,view,ground_z,pose,timestamp,goal,present):
        if not present:raise ValueError(f'{self.client.name} is a point-goal planner; exploration needs CurveNav')
        rgb,depth=scene.raw_observation(self.sim,view,ground_z=ground_z,rgb=self.rgb)
        if rgb is None:rgb=np.zeros(depth.shape+(3,),np.uint8)
        yaw=float(view[2])
        state=dict(robot_pos=pose[:3,3],robot_quat=[0.,0.,np.sin(yaw/2),np.cos(yaw/2)],body_to_world=pose,
            camera_to_body=scene.sensor_assets()[2],camera_intrinsics=self.k,timestamps=float(timestamp))
        trajectory=self.client.step(rgb,depth,(np.asarray(goal,np.float64)-self.origin).astype(np.float32),state)
        view=np.asarray(view).copy()
        def complete():
            c,s=np.cos(view[2]),np.sin(view[2])
            for key,value in zip(self.decisions,(view,float(timestamp),trajectory)):self.decisions[key].append(value)
            return np.vstack([view[:2],view[:2]+(self.origin+trajectory)@np.array([[c,s],[-s,c]])]),{}
        return complete

    def finish(self):
        trajectories=self.decisions.pop('trajectories')
        width=max((len(t) for t in trajectories),default=0)  # servers may vary their horizon; pad with the last point
        return dict({key:np.asarray(value) for key,value in self.decisions.items()},
            trajectories=np.asarray([np.vstack([t,np.repeat(t[-1:],width-len(t),0)]) for t in trajectories]))


def run_episode(planner,case,seconds,*,controller,navigation,latency_s,simulator=None):
    """One closed-loop episode on simulation time: each plan is computed from the state at its observation and
    published latency_s later, while the controller keeps tracking the previous plan at its own period. Results
    do not depend on CPU load or concurrency."""
    runtime,_,_=scene.sensor_assets()
    sim=scene.create_simulator(Path(case['assets']),case['asset_scene_id'],0,runtime['camera'],case['scene_dataset'],planner.rgb) if simulator is None else simulator
    planner.start(sim);plans=[];motion=None
    route=Subgoals(case['route'],case.get('route_doubtful') or ()) if case.get('route') else None  # global route: the policy is given its subgoals
    def infer(view,timestamp):
        ground_z=navigation.ground_height(view[:2])
        pose=scene.body_pose(view,ground_z=ground_z)
        present=case['goal_present']
        target=route.target(pose[:2,3]) if route is not None and present else case['goal']
        goal=((np.asarray(target)-pose[:2,3])@pose[:2,:2]).astype(np.float32) if present else np.zeros(2,np.float32)
        return planner.request(view,ground_z,pose,timestamp,goal,present)
    try:
        with torch.inference_mode():
            view=np.asarray(case['start'],dtype=np.float64)
            world,record=infer(view,0.)()
            motion=RealtimeMotion(view,navigation.body_clearance,seconds,case['goal'] if case['goal_present'] else None,controller=controller,
                collision_ends=case.get('collision_ends',True),arrival_hold_s=case.get('arrival_hold_s') if case['goal_present'] else None)
            publication=motion.publish(world,0.)
            plans.append(dict(observation_time_s=0.,**record,**publication))
            steps=decision_steps(latency_s,controller.period)
            state=motion.snapshot()
            pending=infer(state['view'],state['time_s']) if state['status']=='running' else None
            while pending is not None:
                # Through the latency the controller tracks the previous plan, so the state at this plan's
                # publication, and with it the next observation and request, does not depend on this plan: both
                # run while the model computes it. Every result is the same as observing after publishing.
                motion.control_steps(steps);following=motion.snapshot()
                upcoming=infer(following['view'],following['time_s']) if following['status']=='running' else None
                world,record=pending()
                if motion.status=='controller_failure':break
                publication=motion.publish(world,state['time_s'],state['view'])
                plans.append(dict(observation_time_s=state['time_s'],**record,
                    decision_latency_ms=1000*(motion.time-state['time_s']),**publication))
                state,pending=following,upcoming
        final=motion.snapshot()
        result=dict(status=final['status'],failure_reason=final['failure_reason'],seconds=final['time_s'],path_m=final['path_m'],final_view=final['view'].tolist(),
            contacts=motion.contacts,
            mean_decision_latency_ms=mean_decision_latency_ms(plans),
            plans=plans,motion_trace=motion.trace,controller=controller.metadata)
        return result,planner.finish()
    finally:
        if simulator is None:sim.close()


def evaluate_episode(planner,row,episode,navigation,controller,folder,*,latency_s,exploration=False,simulator=None):
    """Candidate statistics (trajectory_metrics) exist for planners that publish their candidates (CurveNav)."""
    folder.mkdir(parents=True)
    case=dict(row,start=episode['start'],goal=episode['goal'],route=episode.get('route'),route_doubtful=episode.get('route_doubtful'),collision_ends=episode.get('collision_ends',True),arrival_hold_s=episode.get('arrival_hold_s'))
    goal,raw=run_episode(planner,dict(case,goal_present=True),episode['goal_budget_s'],controller=controller,navigation=navigation,simulator=simulator,latency_s=latency_s)
    goal_trajectory=trajectory_metrics(navigation,raw,goal['plans'],torch.device('cpu')) if 'candidate_paths' in raw else None
    np.savez_compressed(folder/'goal_candidates.npz',**raw)
    del raw
    result=dict(episode_id=episode['episode_id'],source=row['source'],scene_id=row['scene_id'],range_m=episode.get('range_m'),
        goal=goal_metrics(goal['status']=='goal_reached',episode['spl_distance_m'] if 'spl_distance_m' in episode else episode['shortest_path_m'],goal['path_m']),goal_trajectory=goal_trajectory,
        goal_collision=goal['status']=='collision' or goal['contacts']>0,goal_contacts=goal['contacts'],
        goal_status=goal['status'],goal_failure_reason=goal['failure_reason'],
        goal_decision_latency_ms=goal['mean_decision_latency_ms'])
    save_execution(folder,'goal',goal)
    if exploration:
        expert=run_exploration_expert(row,episode,navigation,controller,simulator=simulator)
        save_execution(folder,'expert',expert)
        exploring,raw=run_episode(planner,dict(case,goal_present=False),episode['exploration_budget_s'],controller=controller,navigation=navigation,simulator=simulator,latency_s=latency_s)
        exploration_trajectory=trajectory_metrics(navigation,raw,exploring['plans'],torch.device('cpu'))
        np.savez_compressed(folder/'exploration_candidates.npz',**raw)
        _,k,camera=scene.sensor_assets();curve=coverage_curve(navigation,raw,k,camera)
        scores=exploration_metrics(curve.times,curve.areas,expert['times'],expert['areas'],episode['reachable_area_m2'],
            exploring['status']=='collision',initial_area_m2=curve.initial_area_m2,expert_collision=expert['collision'],
            expert_failure=expert['status'] in ('controller_failure','planning_failure'),failure=exploring['status']=='controller_failure')
        result.update(exploration=scores,exploration_trajectory=exploration_trajectory,exploration_collision=exploring['status']=='collision',
            exploration_status=exploring['status'],expert_status=expert['status'],expert_collision=expert['collision'],
            exploration_failure_reason=exploring['failure_reason'],expert_failure_reason=expert['failure_reason'])
        save_execution(folder,'exploration',exploring)
        np.savez_compressed(folder/'exploration_curve.npz',times=curve.times,areas=curve.areas,expert_times=expert['times'],expert_areas=expert['areas'])
    write(folder/'result.json',result)
    print(json.dumps(result),flush=True)
    return result


def expected_seconds(episodes,hint=None):
    """Expected simulated duration of each episode, used only to schedule the longest first (no result depends on
    it): an earlier run's measured goal duration where given, otherwise the budget scaled by the shortest path."""
    measured={}
    if hint is not None:
        for path in Path(hint).glob('worker_*/**/goal_execution.json'):
            measured[json.loads((path.parent/'result.json').read_text())['episode_id']]=json.loads(path.read_text())['seconds']
    distance={e['episode_id']:e.get('spl_distance_m',e.get('shortest_path_m',0.)) for e in episodes}
    longest=max(distance.values(),default=0.) or 1.
    return {e['episode_id']:measured.get(e['episode_id'],e['goal_budget_s']*distance[e['episode_id']]/longest) for e in episodes}


def evaluate_worker(worker,args,connection,render_connection,queue):
    """Episodes come from one shared longest-first queue, so the workers finish together. An episode does not
    depend on which worker runs it or what that worker ran before (the controller and history reset per episode).
    `connection`: the CurveNav model server's pipe, or the port of this worker's baseline server."""
    with (args.output/f'worker_{worker}.log').open('x') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        from kappanav.evaluation.benchmark.shared_model import ACTOR, DATA, NAVIGATIONS
        from kappanav.evaluation.benchmark.inference import RemoteActor, RemoteSimulator
        data=DATA;episodes={e['episode_id']:e for e in data['episodes']}
        torch.set_num_threads(1);scene.configure(args.runtime)
        if args.policy=='curvenav':actor=RemoteActor(ACTOR,connection,render_connection);planner=CurveNavPlanner(actor)
        else:
            from kappanav.evaluation.benchmark.baselines import BaselineClient
            actor=None;planner=BaselinePlanner(BaselineClient(args.policy,connection))
        rows=[]
        folder=args.output/f'worker_{worker}';folder.mkdir()
        controller=(PathMPC if args.controller=='forward' else ReferenceMPC)(args.output/'mpc_codegen',generate=False)
        scenes={r['scene_id']:r for r in data['scenes']}
        for episode_id in iter(queue.get,None):
            episode=episodes[episode_id];row=scenes[episode['scene_id']]
            # The shared renderer owns and keeps the scenes; a remote handle only names one.
            simulator=RemoteSimulator(render_connection,row['scene_id']);navigation=NAVIGATIONS[episode['scene_id']]
            rows.append(evaluate_episode(planner,row,episode,navigation,controller,folder/episode['episode_id'],exploration=args.exploration,simulator=simulator,
                latency_s=args.decision_latency_s))
        (folder/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
        if actor is not None:actor.close()
        render_connection.send(('close',None));render_connection.close()


def main():
    from kappanav.evaluation.benchmark.baselines import BASELINES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=('curvenav',*BASELINES),default='curvenav',
        help='CurveNav, or a baseline planner served from baselines/ (evaluation.baselines; its released checkpoint)')
    parser.add_argument('--checkpoint',type=Path,default=Path('data/model/policy.pt'),help='CurveNav policy bundle (default: the deployed model)')
    parser.add_argument('--benchmark',type=Path,default=Path('data/benchmark/tasks.json'))
    parser.add_argument('--maps',default='none',help="global route map: 'none' (the policy is given the goal), 'oracle' (the "
        "task's C-space shortest path) or a directory of predicted <scene_id>.npz (global_planner.route)")
    parser.add_argument('--runtime',type=Path,default=scene.DEFAULT_RUNTIME)
    parser.add_argument('--output',type=Path,help='new output directory (default outputs/evaluation/<UTC time>)')
    parser.add_argument('--gpus',type=int,nargs='+',help='physical GPU ids (default CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--workers-per-gpu',type=int,default=8,help='Independent environment processes served by each GPU')
    parser.add_argument('--controller',choices=('reference','forward'),default='reference',help='reference benchmark MPC or the forward-only deployment PathMPC')
    parser.add_argument('--exploration',action='store_true',help='also run each episode without a goal against the exploration expert (E-SPL)')
    parser.add_argument('--servers-per-gpu',type=int,help='model processes per GPU; workers are spread over them '
        '(default half the workers: on one GPU with 8 workers, 4 beat 2 and 8, whose extra CUDA contexts the GPU time-slices)')
    parser.add_argument('--renderers-per-gpu',type=int,help='renderer (and frame projection) processes per GPU; default servers-per-gpu')
    parser.add_argument('--decision-latency-s',type=float,default=DECISION_LATENCY_S,
        help='simulation time from an observation to its plan being published')
    parser.add_argument('--inference-batch-size',type=int,default=1,
        help='requests per model batch. BF16 decisions depend on the batch composition, which depends on timing, so only 1 '
        'makes every episode reproducible (bitwise, for any worker, server or GPU count); 8 runs ~20%% faster')
    parser.add_argument('--cuda-graphs',action=argparse.BooleanOptionalAction,default=True,
        help='replay candidate generation from CUDA graphs (inference.GraphedCandidates; bitwise the eager result)')
    parser.add_argument('--schedule-from',type=Path,help='an earlier run output directory whose episode durations order the queue (scheduling only)')
    args=parser.parse_args()
    # A terminated run still stops its model, renderer and worker processes (evaluate's finally), which would
    # otherwise keep the GPU.
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    from kappanav.local_planner.control.mpc import MPC_PERIOD_S
    decision_steps(args.decision_latency_s,MPC_PERIOD_S)
    if args.gpus is None:args.gpus=[int(gpu) for gpu in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if gpu] or list(range(torch.cuda.device_count()))
    if args.output is None:args.output=Path('outputs/evaluation')/time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    if args.servers_per_gpu is None:args.servers_per_gpu=max(1,args.workers_per_gpu//2)
    if args.renderers_per_gpu is None:args.renderers_per_gpu=args.servers_per_gpu
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,args.gpus))
    args.rgb=args.policy!='curvenav' and BASELINES[args.policy].rgb  # the renderers add a colour sensor
    if args.exploration and args.policy!='curvenav':parser.error('exploration is evaluated for CurveNav only')
    summary=evaluate(args)
    print(json.dumps(summary['groups']['all']),flush=True)


def evaluate(args):
    """One complete pass over the benchmark, with shared model, renderer and worker processes."""
    from kappanav.common.physical import ROBOT
    benchmark=json.loads(args.benchmark.read_text())
    if benchmark.get('robot',ROBOT.name)!=ROBOT.name:
        raise ValueError(f'benchmark robot {benchmark["robot"]} does not match selected {ROBOT.name}')
    if args.maps!='none':benchmark=plan_routes(benchmark,args.maps)
    workers=len(args.gpus)*args.workers_per_gpu
    args.output.mkdir(parents=True,exist_ok=False)
    tasks=args.output/'tasks.json';tasks.write_text(json.dumps(benchmark)+'\n')  # the episodes as run, routes included
    (PathMPC if args.controller=='forward' else ReferenceMPC)(args.output/'mpc_codegen')
    package=Path(__file__).resolve().parents[2]
    from kappanav.evaluation.benchmark import baselines
    checkpoint=args.checkpoint if args.policy=='curvenav' else baselines.WEIGHTS/baselines.BASELINES[args.policy].checkpoint
    identity=dict(robot=ROBOT.name,policy=args.policy,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),benchmark_sha256=sha(args.benchmark),maps=args.maps,runtime_sha256=sha(args.runtime),
        source_sha256={str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*.py'))},backend='habitat',controller=args.controller,gpus=args.gpus,parallel_workers=workers,exploration=args.exploration,decision_latency_s=args.decision_latency_s,execution='shared_model_and_renderer_per_gpu',servers_per_gpu=args.servers_per_gpu,renderers_per_gpu=args.renderers_per_gpu,inference_batch_size=args.inference_batch_size,cuda_graphs=args.cuda_graphs)
    (args.output/'inputs.json').write_text(json.dumps(dict(identity,execution_order=EXECUTION_ORDER),indent=2)+'\n')
    os.environ['KAPPANAV_EVAL_INPUTS']=json.dumps([str(args.checkpoint.resolve()) if args.policy=='curvenav' else None,str(tasks.resolve())])
    mp.set_forkserver_preload(['kappanav.evaluation.benchmark.shared_model'])
    from kappanav.evaluation.benchmark.inference import serve, serve_render
    context=mp.get_context('forkserver');links=[context.Pipe() for _ in range(workers)];renders=[context.Pipe() for _ in range(workers)];jobs=[]
    episodes=benchmark['episodes'];expected=expected_seconds(episodes,args.schedule_from)
    queue=context.Queue()
    for episode in sorted(episodes,key=lambda e:(-expected[e['episode_id']],e['episode_id'])):queue.put(episode['episode_id'])
    for _ in range(workers):queue.put(None)
    servers=[]
    try:
        # Worker w uses GPU w % gpus and, on it, model server (w // gpus) % servers-per-gpu and renderer
        # (w // gpus) % renderers-per-gpu: one process serialises its workers, so several per GPU keep the device
        # busy, while every extra process is one more CUDA context the GPU time-slices. A baseline has one policy
        # server per worker instead (its history is per episode).
        if args.policy!='curvenav':
            ports=[baselines.free_port() for _ in range(workers)]
            for worker,port in enumerate(ports):
                log=(args.output/f'baseline_{worker}.log').open('x')
                servers.append(baselines.start_server(args.policy,args.gpus[worker%len(args.gpus)],port,log));log.close()
            for process,port in zip(servers,ports):baselines.wait_ready(process,port)
        for kind,target,pipes,count in ((('inference',serve,links,args.servers_per_gpu),) if args.policy=='curvenav' else ())+(('render',serve_render,renders,args.renderers_per_gpu),):
            groups=len(args.gpus)*count
            for group in range(groups):
                gpu=args.gpus[group%len(args.gpus)];server=group//len(args.gpus)
                process=context.Process(target=target,args=(gpu,args,[pipes[w][1] for w in range(group,workers,groups)],server),name=f'{kind}_{gpu}_{server}')
                process.start();jobs.append(process)
        for _,connection in links+renders:connection.close()
        for worker,(connection,_) in enumerate(links):
            process=context.Process(target=evaluate_worker,args=(worker,args,connection if args.policy=='curvenav' else ports[worker],renders[worker][0],queue),name=f'worker_{worker}')
            process.start();jobs.append(process);connection.close();renders[worker][0].close()
        pending={process.sentinel:process for process in jobs}
        while pending:
            for sentinel in wait(pending):
                process=pending.pop(sentinel);process.join()
                if process.exitcode:raise RuntimeError(f'{process.name} exited with {process.exitcode}; inspect logs')
    finally:
        for process in jobs:
            if process.is_alive():process.terminate()
        for process in jobs:process.join()
        for process in servers:process.terminate()
        for process in servers:process.wait()
    rows=[row for index in range(workers) for row in json.loads((args.output/f'worker_{index}/rows.json').read_text())]
    from kappanav.evaluation.benchmark.results import write_summary
    return write_summary(args.output,rows,episodes,execution_order=EXECUTION_ORDER,**identity)


if __name__=='__main__':main()
