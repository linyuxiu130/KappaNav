"""In-memory capture supervision and the train-only normalization of a fresh model."""
from pathlib import Path
import argparse,hashlib,io,json
import numpy as np
import torch
import yaml
from kappanav.local_planner.model.config_io import config_from_mapping
from kappanav.common.experiment_io import sha
from kappanav.local_planner.data.bspline_teachers import (exploration_targets, exploration_teachers,
                                                          qualified_route_teacher, reachable_cells, relabelled_goals,
                                                          source_utility_availability, unpack_routes)
from kappanav.common.geometry import Grid
from kappanav.local_planner.inference.bundle import create_bundle, save_bundle
from kappanav.local_planner.features.conditions import live_conditions

# One second of 0.2 s anchors shares its extra teachers' fitting cost (see label_records).
EXTRA_TEACHER_EVERY=5
# An anchor within this motion of the last labelled one is the same state (see label_records).
STILL_POSITION_M,STILL_HEADING_RAD=.02,.05


def same_state(view,labelled):
    """An anchor within STILL_* of the last labelled one repeats its state (the robot stood still between them);
    label_records labels it once."""
    return labelled is not None and np.linalg.norm(view[:2]-labelled[:2])<STILL_POSITION_M and abs(np.angle(np.exp(1j*(view[2]-labelled[2]))))<STILL_HEADING_RAD


@torch.no_grad()
def label_records(actor,folder,*,raw,utility,helpers=None):
    """Build one capture's supervision in memory, sharing geometry between its goal modes.

    The anchors' CPU geometry (local_labels, exploration rankings and routes, relabelled goals' expert routes)
    is computed by `helpers` (capture_helpers.Helpers) when given, else here; the same functions on the same
    inputs either way, so the records do not depend on it."""
    from kappanav.local_planner.data import capture_helpers
    folder=Path(folder);raw_path=folder/'capture.json'
    raw=dict(raw,_depth_path=str((folder/'depths.npy').resolve()))
    digest=sha(raw_path);grid=folder/'source_grid.npz';grid_sha=sha(grid);planner=Grid.load(grid)
    stride=int(raw['anchor_stride']);routes=unpack_routes(raw);history=actor.new_history();anchors=[]
    rng=np.random.default_rng(int(digest[:16],16));labelled=None
    context=capture_helpers.label_context(actor)
    run=capture_helpers.Inline(actor,planner,utility,raw) if helpers is None else helpers
    navigation=str(Path(raw['case']['navigation']).resolve())
    # Only captures in which the expert reached the goal teach: a failed drive (budget, collision) repeats the state
    # it got stuck in and its routes were not executable from there.
    for index in range(raw['eligible_anchor_count'] if raw['positive_expert'] else 0):
        history.update(raw['depths'][index],raw['camera_intrinsics'][index],raw['camera_to_body'][index],raw['body_to_world'][index],float(raw['timestamps'][index]))
        selection=history.sample()
        if index%stride:continue
        history.plan_write(selection);plan=history.read_plan();plan['raw_sha256']=digest
        view=raw['acquired_views'][index]
        # A robot standing still repeats the same state for several anchors; one labelled anchor per pose is
        # enough. In-place turning changes the heading and is always labelled.
        if same_state(view,labelled):
            continue
        labelled=np.asarray(view,dtype=np.float64)
        stream=io.BytesIO();torch.save(plan,stream)
        plan_sha=hashlib.sha256(stream.getbuffer()).hexdigest()
        # Neighbouring anchors (0.2 s apart) see nearly the same observation; the extra no-goal and goal teachers
        # are fitted once per EXTRA_TEACHER_EVERY anchors, the recorded goal teacher at every anchor.
        # The relabelled goals consume the capture's rng in anchor order; one grid search serves them and the
        # exploration targets.
        extra_anchor=(index//stride)%EXTRA_TEACHER_EVERY==0
        reach=reachable_cells(planner,view) if extra_anchor else None
        targets=exploration_targets(actor,planner,view,reach) if extra_anchor else None
        goals=relabelled_goals(planner,view,rng,reach) if extra_anchor else ()
        selected=sorted(set(plan['selection']['indices']))
        anchors.append(dict(index=index,view=view,plan=plan,plan_sha=plan_sha,goals=goals,
            local=run.label(capture_helpers.anchor_geometry,context,str(folder.resolve()),navigation,selected,view,targets),
            relabel=[run.label(capture_helpers.goal_route,navigation,view,world) for world in goals]))
    records=[]
    for anchor in anchors:
        index,view,plan,plan_sha=anchor['index'],anchor['view'],anchor['plan'],anchor['plan_sha']
        field,tree,error,ranking,explored=anchor['local'].result()
        pose=raw['body_to_world'][index];route=routes[index//stride]
        teacher=None if route is None else qualified_route_teacher(actor,view,pose,route,tree,error,utility.source_query)[0]
        goal=((np.asarray(raw['case']['goal'])-pose[:2,3])@pose[:2,:2]).astype(np.float32)
        common=dict(step=index,view=view,acquisitions=index+1,
            teacher_values=None if teacher is None else teacher['teacher_values'],
            trial_id=raw['provenance']['trial_id'],raw_path=str(raw_path.resolve()),raw_sha256=digest,
            history_plan=plan,history_plan_path=f'{digest}:anchor:{index}',history_plan_sha256=plan_sha,
            source_grid_sha256=grid_sha,source_grid_index=0,
            local_target_field=field,distance_label_available=tree.n>0)
        # Without a mission the teacher is not the hidden goal's route: it is the best certified exploration
        # route under the no-goal utility Q learns (one row per teacher; a teacherless row keeps LH/LQ).
        explore=[] if ranking is False else exploration_teachers(actor,planner,view,pose,tree,error,utility,field,ranking=ranking,routes=explored)
        rows=[(True,goal,common['teacher_values'])]+[(False,np.zeros(2,np.float32),t['teacher_values']) for t in explore]
        # Extra goals for the same observation, each with its own expert teacher (rows without one are dropped).
        for world,extra in zip(anchor['goals'],anchor['relabel'],strict=True):
            extra=extra.result()
            extra=None if extra is None else qualified_route_teacher(actor,view,pose,extra,tree,error,utility.source_query)[0]
            if extra is not None:rows.append((True,((world-pose[:2,3])@pose[:2,:2]).astype(np.float32),extra['teacher_values']))
        if not explore:rows.append((False,np.zeros(2,np.float32),None))
        for present,point,values in rows:
            item=dict(common,goal_present=present,goal=torch.from_numpy(point),teacher_values=values)
            item['q_supervision_available'],_=source_utility_availability(utility.goal_distance,view,item['goal'])
            records.append((item,raw,None,utility))
    return records


def _array_teachers(path):
    """Read only the two small row columns needed for train normalization."""
    from kappanav.local_planner.data.array_store import ArrayTable
    scene=json.loads(path.read_text())
    if scene['split']!='train' or not scene['rows']:return
    spec=json.loads((path.parent/'manifest.json').read_text())['tables']['rows']
    fields={key:spec['schema']['fields'][key] for key in ('teacher_values','goal_present')}
    names=set()
    for node in fields.values():
        while node['kind']=='optional':
            names.add(node['index']);node=node['child']
            if node is None:break
        if node is not None:names.update(node[key] for key in ('values','offsets','shapes') if key in node)
    table=ArrayTable(path.parent/'rows',dict(spec,schema=dict(kind='dict',fields=fields),
        files={name:spec['files'][name] for name in names}))
    for index in range(len(table)):
        if table.field(index,'goal_present'):
            value=table.field(index,'teacher_values')
            if value is not None:yield value


def dataset_scenes(root):
    """Published dataset membership, or completed scene markers before index publication."""
    root=Path(root);dataset=root/'dataset.json'
    if dataset.is_file():
        return [(dataset.parent/entry['path']).resolve() for entry in json.loads(dataset.read_text())['scenes']]
    return sorted(root.glob('shards/*/scene.json'))


def fit_statistics(roots,config_path,bundle_path):
    """Train-only normalization -> a fresh bundle; existing compiled snapshots stay immutable."""
    if bundle_path.exists():raise FileExistsError(bundle_path)
    values=[value for root in roots for path in dataset_scenes(root) for value in _array_teachers(path)]
    if not values:raise ValueError('No qualified training teachers found')
    controls=torch.stack(values).reshape(-1,7,2)
    increments=torch.diff(torch.cat((torch.zeros(len(controls),1,2),controls),1),dim=1).flatten(1)
    mean=increments.mean(0);std=increments.std(0,unbiased=False)
    config=yaml.safe_load(config_path.read_text())
    config['model']['trajectory'].update(control_increment_mean_xy_m=mean.tolist(),control_increment_std_xy_m=std.tolist())
    torch.manual_seed(42);actor=create_bundle(config_from_mapping(config))
    with bundle_path.open('xb') as stream:save_bundle(stream,base_config=config,policy=actor.policy,head=actor.head,architecture=actor.architecture,
        provenance=dict(initialization='fresh_all_parameters',normalization='qualified train teachers',normalization_rows=len(values),
            normalization_roots=[str(root.resolve()) for root in roots],seed=42))
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))


@torch.no_grad()
def record_candidate_banks(actor,records):
    """Attach one capture's fixed native candidates, using the original batch order."""
    device=next(actor.policy.parameters()).device
    if not records:return
    item,raw,*_=records[0];key=item['raw_sha256']
    ids=sorted({int(index) for item,*_ in records for index in item['history_plan']['frame_ids']})
    maps={}
    for start in range(0,len(ids),64):
        indices=ids[start:start+64]
        depth=torch.as_tensor(raw['depths'][indices],device=device,dtype=torch.float32)[:,None]
        features=actor.policy.depth_encoder.image_features(depth)
        maps.update({(key,index):feature[None] for index,feature in zip(indices,features,strict=True)})
    for start in range(0,len(records),16):
        batch=records[start:start+16]
        condition,_=live_conditions(actor,batch,maps)
        paths=actor.native_bank(condition)['paths'].float().cpu()
        for (item,*_),path in zip(batch,paths,strict=True):item['critic_replay_paths']=path.clone()



def main():
    parser=argparse.ArgumentParser(description='Fit the train-only control-increment normalization of a fresh model.')
    parser.add_argument('--root',type=Path,nargs='+',required=True,help='native dataset roots')
    parser.add_argument('--config',type=Path,default=Path('configs/base.yaml'))
    parser.add_argument('--bundle',type=Path,required=True,help='fresh output model')
    args=parser.parse_args();torch.set_num_threads(1)
    fit_statistics(args.root,args.config,args.bundle)


if __name__=='__main__':main()
