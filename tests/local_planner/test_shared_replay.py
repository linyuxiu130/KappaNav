"""The canonical array snapshot opens once before CUDA and isolates rank writes."""
import os,subprocess,sys,textwrap
from pathlib import Path
from kappanav.local_planner.data import bspline_dataset
from test_replay_pack import make_array_replay


def test_forkserver_opens_arrays_once_and_isolates_worker_writes(tmp_path):
    _,manifest,_,_=make_array_replay(tmp_path/'data');calls=tmp_path/'loads.jsonl'
    (tmp_path/'preload_fixture.py').write_text(textwrap.dedent(f'''
        import os,json,torch
        from pathlib import Path
        from test_replay_pack import array_actor
        from kappanav.local_planner.inference import bundle as bspline_bundle
        from kappanav.local_planner.data import replay_pack
        bspline_bundle.load_bundle=lambda *args:array_actor()
        original=replay_pack.load_array_replay
        def opened(*args,**kwargs):
            with Path({str(calls)!r}).open('a') as stream:stream.write(json.dumps(dict(pid=os.getpid()))+'\\n')
            return original(*args,**kwargs)
        replay_pack.load_array_replay=opened
        def forbidden(*args,**kwargs):raise AssertionError('Legacy tensor cache read during array preload')
        torch.load=forbidden
        import kappanav.local_planner.training.shared_replay
    '''))
    script=tmp_path/'probe.py';script.write_text(textwrap.dedent(f'''
        import json,os,multiprocessing as mp
        import torch
        def worker(rank,queue,barrier):
            from kappanav.local_planner.training.shared_replay import TRAINING,LOADER_PID,SOURCE_KEYS
            assert TRAINING[0][0][0]['q_expert_paths'][0,0,0].item()==7.
            raw=TRAINING[0][0][1]['depths'];before=float(raw.flat[0])
            if rank==0:raw.flat[0]=3.
            barrier.wait()
            queue.put(dict(rank=rank,pid=os.getpid(),loader=LOADER_PID,
                address=raw.__array_interface__['data'][0],before=before,after=float(raw.flat[0]),
                cuda=torch.cuda.is_initialized(),same_sources=SOURCE_KEYS[0]==SOURCE_KEYS[1]))
        if __name__=='__main__':
            os.environ['KAPPANAV_SHARED_REPLAY_INPUTS']=json.dumps(['bundle',[{str(manifest)!r}],[]])
            mp.set_forkserver_preload(['preload_fixture'])
            context=mp.get_context('forkserver');queue=context.Queue();barrier=context.Barrier(2);loaders=set()
            for _ in range(2):
                torch.multiprocessing.start_processes(worker,args=(queue,barrier),nprocs=2,start_method='forkserver')
                rows=sorted([queue.get(timeout=5) for _ in range(2)],key=lambda x:x['rank'])
                loaders.update(r['loader'] for r in rows)
                assert len(loaders)==1 and rows[0]['address']==rows[1]['address']
                assert all(r['loader']!=r['pid'] and not r['cuda'] and r['same_sources'] for r in rows)
                assert rows[0]['before']==rows[1]['before']==0.
                assert rows[0]['after']==3. and rows[1]['after']==0.
    '''))
    source=Path(bspline_dataset.__file__).resolve().parents[3];tests=os.pathsep.join(map(str,(Path(__file__).parent,Path(__file__).parents[1])))
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(tmp_path),str(tests),str(source),os.environ.get('PYTHONPATH','')]))
    subprocess.run([sys.executable,str(script)],env=env,check=True,capture_output=True,text=True,timeout=60)
    assert len(calls.read_text().splitlines())==1
