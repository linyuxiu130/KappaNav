"""Baseline local planners in CurveNav's closed loop.

Each baseline (baselines/<name>, vendored with its licence and provenance in baselines/README.md) runs as its own
policy server and speaks the shared raw-array HTTP protocol (baselines/serving/protocol.py). The evaluator gives it
what it gives CurveNav (the same tasks, subgoals, camera, decision latency, controller, robot body and contact):
the rendered colour and depth at full resolution with their intrinsics, the robot state, and the point goal in the
planar frame the method was evaluated in upstream (x forward, y left, at the camera or at the robot base; Baseline
origin). The server answers with a trajectory in that frame, which the controller tracks. One server per
evaluation worker keeps every episode's policy history to itself.
"""
from dataclasses import dataclass
import io,json,os,socket,subprocess,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'baselines'
WEIGHTS=Path(os.environ.get('KAPPANAV_BASELINE_WEIGHTS','/shibo_huang/data/curvenav/weights'))
PYTHON=os.environ.get('KAPPANAV_BASELINE_PYTHON',sys.executable)
STOP_THRESHOLD=-3.  # the critic value below which NavDP-family servers stop (the released benchmark's)
SEED=1234           # policy-side sampling seed at every episode start (the released benchmark's)


@dataclass(frozen=True)
class Baseline:
    server: tuple       # server arguments after the Python executable; *.py and *.yaml are relative to ROOT
    checkpoint: str     # relative to WEIGHTS
    rgb: bool           # whether the planner reads colour (depth-only planners are sent a black image)
    cwd: str='.'        # relative to ROOT
    path: tuple=()      # PYTHONPATH entries besides ROOT, relative to ROOT
    origin: str='camera'  # where its goal and trajectory frame sits: 'camera' (NavDP's benchmark plans from the camera
                          # pose) or 'base' (X-NavDP's own benchmark: the robot base); the heading is the body's


BASELINES={
    'sandplanner':Baseline(('sandplanner/sand_planner/server/simple_server.py',),'sandplanner/NoMax.pth',False,path=('sandplanner',)),
    'iplanner':Baseline(('iplanner/iplanner_server.py','--config','iplanner/configs/iplanner.yaml'),'iplanner/plannernet.pt',False),
    'navdp':Baseline(('navdp/navdp_server.py',),'navdp/navdp_pretrain.ckpt',True),
    'x-navdp':Baseline(('-m','eval.src.policy_server','--embodiment','wheeled'),'x-navdp/x-navdp_posttrain.ckpt',True,cwd='x-navdp',path=('x-navdp',),origin='base'),
}


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));return probe.getsockname()[1]


def nvrtc_library():
    """The CUDA runtime compiler shipped with torch (12.8): CuPy compiles its kernels with it, and an older one found
    first on the library path cannot target the Blackwell GPUs (sm_120)."""
    import nvidia.cuda_nvrtc
    return str(Path(nvidia.cuda_nvrtc.__file__).parent/'lib')


def start_server(name,gpu,port,log):
    """The policy server of baseline `name` on physical GPU `gpu`, listening on `port`; output goes to `log`."""
    baseline=BASELINES[name]
    checkpoint=WEIGHTS/baseline.checkpoint
    if not checkpoint.is_file():raise FileNotFoundError(f'{name} checkpoint missing: {checkpoint}')
    arguments=[str(ROOT/a) if a.endswith(('.py','.yaml')) else a for a in baseline.server]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=os.pathsep.join([str(ROOT),*(str(ROOT/p) for p in baseline.path)]),
        LD_LIBRARY_PATH=os.pathsep.join([nvrtc_library(),os.environ.get('LD_LIBRARY_PATH','')]))
    return subprocess.Popen([PYTHON,*arguments,'--port',str(port),'--device','cuda:0','--checkpoint',str(checkpoint)],
        cwd=ROOT/baseline.cwd,env=env,stdout=log,stderr=subprocess.STDOUT)


def wait_ready(process,port,timeout_s=900):
    import requests
    deadline=time.time()+timeout_s
    while time.time()<deadline:
        if process.poll() is not None:raise RuntimeError(f'baseline server on port {port} exited with {process.returncode}')
        try:
            if requests.get(f'http://127.0.0.1:{port}/health',timeout=5).ok:return
        except requests.ConnectionError:pass
        time.sleep(2)
    raise TimeoutError(f'baseline server on port {port} not ready in {timeout_s} s')


class BaselineClient:
    """One worker's connection to its baseline server."""
    def __init__(self,name,port):
        import requests
        self.name,self.rgb,self.origin=name,BASELINES[name].rgb,BASELINES[name].origin
        self.url=f'http://127.0.0.1:{port}';self.session=requests.Session()

    def reset(self,intrinsic):
        """A fresh policy history for a new episode."""
        response=self.session.post(self.url+'/navigator_reset',json=dict(intrinsic=np.asarray(intrinsic).tolist(),
            stop_threshold=STOP_THRESHOLD,batch_size=1,global_batch_size=1,batch_start=0,seed=SEED,sample_indices=[],scene_name=''),timeout=900)
        response.raise_for_status()

    def step(self,rgb,depth,goal,state):
        """The trajectory (N x 2, in the baseline's frame) for colour (H x W x 3 uint8), depth (H x W metres, inf = no
        return), the goal in that frame and the robot state (robot_pos, robot_quat xyzw, body_to_world, camera_to_body,
        camera_intrinsics, timestamps)."""
        arrays=dict(image=np.ascontiguousarray(rgb[None],np.uint8),depth=np.ascontiguousarray(depth[None,...,None],np.float32))
        data=dict(goal_data=json.dumps(dict(goal_x=[float(goal[0])],goal_y=[float(goal[1])])),
            state_data=json.dumps({key:np.asarray(value)[None].tolist() for key,value in dict(state,planning_goal=goal).items()}))
        for key,value in arrays.items():data[f'{key}_shape']=json.dumps(list(value.shape));data[f'{key}_dtype']=value.dtype.str
        files={key:(f'{key}.raw',memoryview(value).cast('B'),'application/octet-stream') for key,value in arrays.items()}
        response=self.session.post(self.url+'/pointgoal_step',files=files,data=data,timeout=900)
        response.raise_for_status()
        with np.load(io.BytesIO(response.content),allow_pickle=False) as payload:trajectory=np.asarray(payload['trajectory'],np.float64)
        trajectory=trajectory.reshape(-1,trajectory.shape[-1])[:,:2]
        if not np.isfinite(trajectory).all():raise ValueError(f'{self.name} returned a non-finite trajectory')
        return trajectory
