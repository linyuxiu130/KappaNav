"""Training scene geometry and the one training task catalog, without evaluation-scene leakage.

Every source (`--source NAME=ASSETS`, in catalog order) gets its geometry under OUTPUT/NAME/navigation/<scene_id>
and its own catalog OUTPUT/tasks_NAME.json; OUTPUT/tasks.json concatenates them. Tasks are seeded by the scene's
index within its source, so adding a source leaves the others' tasks unchanged.
"""
from pathlib import Path
import argparse,json
from kappanav.common.experiment_io import sha
from kappanav.evaluation.simulation import scene
from kappanav.evaluation.simulation.navigation import prepare_scene
from kappanav.evaluation.simulation.tasks import sample_episodes, wall_episodes


def training_scenes(assets):
    """GRScenes keep their official train split (the benchmark uses GRScenes evaluation scenes); every downloaded
    HSSD scene is training-only."""
    if not (assets/'scene_split.json').is_file():
        manifest=assets/'download_manifest.json';rows=[]
        for sid in json.loads(manifest.read_text())['scenes']:
            rows.append(dict(scene_id='train/hssd_'+sid,source='train',domain='home',asset_scene_id=sid,
                assets=str(assets.resolve()),scene_dataset='hssd-hab.scene_dataset_config.json'))
        return rows,sha(manifest)
    split=json.loads((assets/'scene_split.json').read_text());rows=[]
    for domain in ('home','commercial'):
        for sid in split[domain+'_train']:
            rows.append(dict(scene_id='train/'+sid,source='train',domain=domain,
                asset_scene_id=sid,assets=str(assets.resolve()),scene_dataset='grscenes.scene_dataset_config.json'))
    return rows,sha(assets/'scene_split.json')


def source_tasks(prepared,tasks_per_scene,wall_tasks_per_scene):
    """One source's tasks: per scene, official-range tasks and near-wall escape tasks; every tenth scene validates."""
    # Every task keeps its sampled start heading: the expert route (bspline_teachers.expert_route) turns toward
    # the route where no forward curve certifies, so no start is rotated or dropped.
    tasks=[]
    for index,row in enumerate(prepared):
        partition='validation' if index%10==9 else 'train'
        kinds=[(sample_episodes(row,seed=20261101+index,count=tasks_per_scene),())] if tasks_per_scene else []
        if wall_tasks_per_scene:
            kinds.append((wall_episodes(row,seed=20261301+index,count=wall_tasks_per_scene),('start_clearance_m',)))
        for episodes,extra in kinds:
            for episode in episodes:
                tasks.append(dict(case_id=episode['episode_id'].replace('/','_'),trial_id=row['scene_id'],split=partition,
                    case=dict(row,start=episode['start'],goal=episode['goal'],goal_present=True,**{key:episode[key] for key in extra})))
    return tasks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',action='append',required=True,metavar='NAME=ASSETS',help='a GRScenes or HSSD asset root (repeat, in catalog order)')
    parser.add_argument('--output',type=Path,default=Path('data/scenes'))
    parser.add_argument('--worker',type=int);parser.add_argument('--workers',type=int,default=7)
    parser.add_argument('--tasks-per-scene',type=int,default=20)
    parser.add_argument('--wall-tasks-per-scene',type=int,default=10,help='escape tasks starting at or near an obstacle')
    parser.add_argument('--exclude',nargs='*',default=[],help='asset scene ids left out (e.g. scenes with unresolved assets)')
    parser.add_argument('--runtime',type=Path,default=scene.DEFAULT_RUNTIME);args=parser.parse_args()
    sources=[]
    for entry in args.source:
        name,_,assets=entry.partition('=')
        if not name or not assets or name in [s[0] for s in sources]:parser.error(f'--source needs a unique NAME=ASSETS: {entry}')
        rows,split_sha=training_scenes(Path(assets))
        sources.append((name,[row for row in rows if row['asset_scene_id'] not in args.exclude],split_sha))
    args.output.mkdir(parents=True,exist_ok=True)
    if args.worker is not None:
        scene.configure(args.runtime)
        jobs=[(name,row) for name,rows,_ in sources for row in rows]
        for name,row in jobs[args.worker::args.workers]:
            if (args.output/name/'navigation'/row['scene_id']/'scene.json').is_file():continue
            prepare_scene(row,args.output/name/'navigation');print(row['scene_id'],flush=True)
        return
    catalog=[];summary={}
    for name,rows,split_sha in sources:
        prepared=[json.loads((args.output/name/'navigation'/row['scene_id']/'scene.json').read_text()) for row in rows]
        tasks=source_tasks(prepared,args.tasks_per_scene,args.wall_tasks_per_scene)
        (args.output/f'tasks_{name}.json').write_text(json.dumps(dict(complete=True,conditions=tasks,source='official_training_split',
            scenes=len(rows),split_sha256=split_sha,evaluation_overlap=0,available_scene_count=len(rows)),indent=2)+'\n')
        count=lambda kind,scene:sum(t['trial_id']==scene and t['case_id'].rsplit('_',2)[-2]==kind for t in tasks)
        summary[name]=dict(scenes=len(rows),tasks=len(tasks),split_sha256=split_sha,
            scenes_short_of_wall_tasks={r['scene_id']:n for r in prepared if (n:=count('wall',r['scene_id']))<args.wall_tasks_per_scene})
        catalog+=tasks
    (args.output/'tasks.json').write_text(json.dumps(dict(complete=True,conditions=catalog,sources=summary,
        scenes=sum(s['scenes'] for s in summary.values()),evaluation_overlap=0,excluded=args.exclude,
        geometry='navmesh C-space without extra erosion (simulation.navigation.save_navigation_grid)'),indent=2)+'\n')
    print(json.dumps(dict(training_scenes=sum(s['scenes'] for s in summary.values()),tasks=len(catalog),sources=summary)),flush=True)


if __name__=='__main__':main()
