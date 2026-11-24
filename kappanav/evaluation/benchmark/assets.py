"""Convert the 40 official evaluation scenes into Habitat assets (data/benchmark/assets)."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,hashlib
from kappanav.evaluation.simulation.assets import export_geometry


def convert(row):
    scene,domain,source_root,output=row
    source_root=Path(source_root);output=Path(output)
    source=source_root/('internscenes_'+domain)/('scenes_'+domain)/scene/'start_result_navigation.usd'
    metadata=export_geometry(source,source_root/('internscenes_'+domain),output/'stages'/(scene+'.glb'))
    (output/'stages'/(scene+'.stage_config.json')).write_text(json.dumps(dict(render_asset=scene+'.glb',collision_asset=scene+'.glb',up=[0,1,0],front=[0,0,-1],units_to_meters=1.)))
    (output/'scenes'/(scene+'.scene_instance.json')).write_text(json.dumps(dict(stage_instance=dict(template_name=scene),object_instances=[])))
    return dict(scene_id=scene,source='benchmark',domain=domain,asset_scene_id=scene,
        assets=str(output),scene_dataset='scenes.scene_dataset_config.json',geometry=metadata,
        source_usd_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='official scene USDs with scene_split.json')
    parser.add_argument('--output',type=Path,default=Path('data/benchmark/assets'))
    parser.add_argument('--workers',type=int,default=7);args=parser.parse_args()
    split=json.loads((args.source/'scene_split.json').read_text());selected=[(sid,domain,str(args.source.resolve()),str(args.output.resolve())) for domain in ('home','commercial') for sid in split[domain+'_eval']]
    assert len(selected)==40 and len({row[0] for row in selected})==40
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'stages').mkdir();(args.output/'scenes').mkdir()
    (args.output/'scenes.scene_dataset_config.json').write_text(json.dumps(dict(stages=dict(paths={'.json':['stages']}),scene_instances=dict(paths={'.json':['scenes']}))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[]
        for row in pool.map(convert,selected):rows.append(row);print(row['scene_id'],flush=True)
    (args.output/'scenes.json').write_text(json.dumps(dict(complete=True,scenes=rows,split_sha256=hashlib.sha256((args.source/'scene_split.json').read_bytes()).hexdigest()),indent=2)+'\n')
