"""One GPU model serves the independent environment processes on that GPU."""
from pathlib import Path
from collections import OrderedDict,deque
from dataclasses import fields,replace
from multiprocessing.connection import wait
from multiprocessing.reduction import ForkingPickler
import os,queue,threading
import numpy as np
import torch
from kappanav.local_planner.features.four_frame_history import (FourFrameHistory, fusion_plan, memory_geometry,
                                                                project_frames)
from kappanav.local_planner.features.history_fusion import fuse_sources_batch
from kappanav.local_planner.features.conditions import encode_conditions


class RemoteHistory(FourFrameHistory):
    """Frames are projected by the GPU renderer process, which is otherwise idle between renders, so a projection
    never queues behind model batches; the model server receives the same frames without a reply."""
    def __init__(self,policy,actor,**kwargs):
        super().__init__(policy,**kwargs);self.actor=actor

    def project(self,frame):
        if frame.index not in self.projections:
            self.actor.projector.send(('project',frame));self.actor.post(('frame',frame))
            self.projections[frame.index]=self.actor.projector.recv()
        return self.projections[frame.index]


class RemoteActor:
    """Requests to the model server are pickled here and written by a sender thread: a request of several hundred
    kB blocks a pipe until the server reads it, and the server reads only between batches, so the episode keeps
    running (and prepares its next request) meanwhile. The server answers each worker's requests in order."""
    def __init__(self,actor,connection,projector):
        self.policy=actor.policy;self.architecture=actor.architecture;self.connection=connection;self.projector=projector
        self.outbox=queue.SimpleQueue();self.sender=threading.Thread(target=self._send,daemon=True);self.sender.start()

    def _send(self):
        for data in iter(self.outbox.get,None):self.connection.send_bytes(data)

    def post(self,message):
        self.outbox.put(bytes(ForkingPickler.dumps(message)))

    def close(self):
        self.post(('close',None));self.outbox.put(None);self.sender.join();self.connection.close()

    def new_history(self):
        return RemoteHistory(self.policy,self,**self.architecture['history'])

    def submit(self,history,goal,goal_present):
        """Send one decision's request; the returned callable waits for its candidate bank, so the caller can run
        work that does not depend on it in between. Requests are answered in the order they are submitted."""
        changed=history.plan_write(history.sample())
        write=fusion_plan(history.cells,changed,history.pose,history.clock,history.selection)
        keys=history._read_keys();lookup={key:i for i,key in enumerate(changed)}
        read=memory_geometry(history.cells,keys,history.pose,history.clock,history.selection)
        indices=np.asarray([lookup[key] for key in keys],dtype=np.int64)
        self.post(('infer',(write,read,indices,goal,goal_present)))
        def receive():
            bank=self.connection.recv()
            return {name:torch.from_numpy(value) if isinstance(value,np.ndarray) else value for name,value in bank.items()}  # 'condition' stays a dict
        return receive

    def step(self,history,goal,goal_present):
        return self.submit(history,goal,goal_present)()


GRAPH_TOKENS=64  # conditions are padded to a multiple of this many tokens, one CUDA graph per padded shape


class GraphedCandidates:
    """actor.candidate_batches replayed from CUDA graphs: one launch instead of ~1800 Python-issued kernels per
    request, which is what bounds a GPU shared by many model processes. Graphs are captured per (batch, token
    count rounded up to GRAPH_TOKENS); a condition is copied into its graph's static input with the extra tokens
    zero and invalid. The bank is bitwise the eager bank of the unpadded condition: invalid tokens are masked out
    exactly (944 padded decisions of six bench episodes, T 336-687, buckets 32-256, all identical, and identical
    closed-loop runs). Captures are kept least-recently-used up to `limit` and share one memory pool."""
    TOKEN_FIELDS=('tokens','token_valid','metric_position','surface_hit','frame_age','motion_token')

    def __init__(self,actor,limit=48):
        self.actor=actor;self.limit=limit;self.pool=torch.cuda.graph_pool_handle();self.graphs=OrderedDict()

    def __call__(self,condition):
        count=condition.tokens.shape[1];padded=-(-count//GRAPH_TOKENS)*GRAPH_TOKENS
        key=(condition.tokens.shape[0],padded)
        if key in self.graphs:
            self.graphs.move_to_end(key);graph,static,bank=self.graphs[key]
            for field in fields(condition):
                value=getattr(condition,field.name);target=getattr(static,field.name)
                if field.name in self.TOKEN_FIELDS:
                    target[:,:count].copy_(value);target[:,count:].zero_()
                else:target.copy_(value)
            graph.replay()
            return bank
        static=replace(condition,**{name:torch.cat((getattr(condition,name),getattr(condition,name).new_zeros(
            (len(condition.tokens),padded-count)+getattr(condition,name).shape[2:])),1) for name in self.TOKEN_FIELDS})
        side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):bank=self.actor.candidate_batches(static,static.terminal_goal)  # warm-up
        torch.cuda.current_stream().wait_stream(side)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,pool=self.pool):captured=self.actor.candidate_batches(static,static.terminal_goal)
        self.graphs[key]=(graph,static,captured)
        while len(self.graphs)>self.limit:self.graphs.popitem(last=False)
        return bank


def serve(gpu,args,connections,server=0):
    with (args.output/f'inference_{gpu}_{server}.log').open('x') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        from kappanav.evaluation.benchmark.shared_model import ACTOR
        torch.set_num_threads(1)
        torch.cuda.set_device(args.gpus.index(gpu))
        actor=ACTOR.to('cuda',non_blocking=True);histories={};active=list(connections);size=args.inference_batch_size
        candidates=GraphedCandidates(actor) if args.cuda_graphs else (lambda condition:actor.candidate_batches(condition,condition.terminal_goal))
        requests=[];batches=deque()
        # A worker may send its next frame and request before its current one is answered. Each worker's messages
        # apply in order and a request holds the later ones until it is answered, so the history the model sees
        # at every request is the same as when a worker waits for each answer.
        inbox={connection:deque() for connection in connections};waiting=set()
        def admit():
            for connection in list(inbox):
                messages=inbox[connection]
                while messages and connection not in waiting:
                    command,payload=messages.popleft()
                    if command=='close':
                        histories.pop(connection,None);del inbox[connection];connection.close();break
                    elif command=='frame':
                        # The renderer projects the frame; the model keeps its depth for image features.
                        frame=payload
                        if frame.index==0:histories[connection]=actor.new_history()
                        histories[connection].update(frame.depth,frame.intrinsic,frame.camera,frame.pose,frame.time)
                    elif command=='infer':requests.append((connection,payload));waiting.add(connection)
                    else:raise ValueError(f'Unknown inference command: {command}')
        # The outer (disabled) autocast region keeps autocast's cache of BF16 weight casts across the model's own
        # autocast regions and requests instead of re-casting every weight per call: the same casts, bit for bit.
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            while inbox:
                ready=batches or requests or any(messages and connection not in waiting for connection,messages in inbox.items())
                for connection in wait(active,timeout=0 if ready else None):
                    message=connection.recv();inbox[connection].append(message)
                    if message[0]=='close':active.remove(connection)
                admit()
                if not batches and requests:
                    requests.sort(key=lambda request:len(request[1][1]['keys']))
                    batches.extend(requests[start:start+size] for start in range(0,len(requests),size))
                    requests=[]
                if batches:
                    group=batches.popleft();plans=[payload[0] for _,payload in group]
                    maps=[];widths=[];pending=[]
                    for connection,(plan,*_) in group:
                        history=histories[connection]
                        for index in np.unique(plan['frame_ids']):
                            index=int(index)
                            if index not in history.maps:pending.append((history,index,history.observations[index].depth))
                        maps.append(history.maps)
                        widths.append({index:frame.depth.shape[1] for index,frame in history.observations.items()})
                    if pending:
                        features=actor.policy.depth_encoder.image_features(torch.as_tensor(np.stack([depth for _,_,depth in pending])[:,None],device='cuda'))
                        for (history,index,_),value in zip(pending,features,strict=True):history.maps[index]=value[None]
                    fused=fuse_sources_batch(actor.policy,plans,maps,widths)
                    visuals=[value.index_select(0,torch.as_tensor(payload[2],device='cuda')) for value,(_,payload) in zip(fused,group,strict=True)]
                    current=encode_conditions(actor,[payload[1] for _,payload in group],visuals,
                        [payload[3] for _,payload in group],[payload[4] for _,payload in group])[0]
                    bank={name:value.cpu().numpy() for name,value in candidates(current).items()}
                    for row,(connection,_) in enumerate(group):
                        connection.send({name:int(value[row]) if name=='selected_index' else value[row] for name,value in bank.items()})
                        waiting.discard(connection)



class RemoteSimulator:
    def __init__(self,connection,scene_id):
        self.connection=connection;self.scene_id=scene_id

    def get_agent(self,index):return self

    def set_state(self,state,*,reset_sensors):
        self.pose=(state.position,state.rotation)

    def get_sensor_observations(self):
        self.connection.send(('render',(self.scene_id,self.pose)))
        return self.connection.recv()

    def close(self):pass  # The shared renderer owns and closes the scene.



def serve_render(gpu,args,connections,server=0):
    with (args.output/f'render_{gpu}_{server}.log').open('x') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        from kappanav.evaluation.benchmark.shared_model import DATA
        from kappanav.evaluation.simulation import scene
        from contextlib import ExitStack
        import habitat_sim
        from kappanav.evaluation.benchmark.shared_model import ACTOR
        scene.configure(args.runtime);runtime,_,_=scene.sensor_assets()
        torch.set_num_threads(1);torch.cuda.set_device(args.gpus.index(gpu))
        policy=None  # moved to the GPU on the first projection
        rows={row['scene_id']:row for row in DATA['scenes']};simulators={};active=list(connections)
        with ExitStack() as resources,torch.inference_mode():
            while active:
                for connection in wait(active):
                    command,payload=connection.recv()
                    if command=='close':active.remove(connection);connection.close();continue
                    if command=='project':
                        if policy is None:policy=ACTOR.policy.to('cuda')
                        connection.send(project_frames(policy,[payload])[0]);continue
                    scene_id,(position,rotation)=payload
                    row=rows[scene_id]
                    asset_key=(row['assets'],row['asset_scene_id'],row['scene_dataset'])
                    if asset_key not in simulators:
                        simulators[asset_key]=scene.create_simulator(Path(row['assets']),row['asset_scene_id'],args.gpus.index(gpu),runtime['camera'],row['scene_dataset'],args.rgb)
                        resources.callback(simulators[asset_key].close)
                    simulator=simulators[asset_key];state=habitat_sim.AgentState()
                    state.position=position;state.rotation=rotation
                    simulator.get_agent(0).set_state(state,reset_sensors=True)
                    connection.send(simulator.get_sensor_observations())
