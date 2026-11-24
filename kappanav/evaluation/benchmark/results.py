"""Complete benchmark summaries and episode-paired comparisons for either simulator.

usage: python -m kappanav.evaluation.benchmark.results RUN [BASELINE]
Each directory contains one summary.json for one pass over its requested task set.
"""
import argparse,json
from pathlib import Path
import numpy as np


def write_summary(folder,rows,episodes,**identity):
    """Publish the common schema in task-file order, without dropping failures or accepting duplicate results."""
    from kappanav.evaluation.benchmark.metrics import summarize
    indexed={row['episode_id']:row for row in rows}
    expected=[episode['episode_id'] for episode in episodes]
    if not expected or len(set(expected))!=len(expected) or len(indexed)!=len(rows) or set(indexed)!=set(expected):
        raise ValueError('Evaluation results must contain every requested episode exactly once')
    ordered=[indexed[key] for key in expected]
    result=dict(complete=all(row['goal_status']!='smoke_limit' for row in ordered),
        groups=summarize(ordered),rows=ordered,**identity)
    (Path(folder)/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def read_summary(run):
    result=json.loads((Path(run)/'summary.json').read_text())
    if not result['complete']:raise ValueError(f'Incomplete evaluation: {run}')
    if len({r['episode_id'] for r in result['rows']})!=len(result['rows']):
        raise ValueError(f'Duplicate episode IDs in {run}/summary.json')
    return result


def paired(run,baseline):
    """Run minus baseline on the exact same tasks; uncertainty is paired across episodes."""
    a,b=({r['episode_id']:r for r in read_summary(folder)['rows']} for folder in (run,baseline))
    if not a or set(a)!=set(b):raise ValueError('Cannot compare different task sets')
    keys=sorted(a)
    x=np.array([a[e]['goal']['success'] for e in keys],float)
    y=np.array([b[e]['goal']['success'] for e in keys],float)
    differences=x-y
    return dict(episodes=len(keys),success_rate_difference=float(differences.mean()),
        standard_error=float(differences.std(ddof=1)/np.sqrt(len(keys))) if len(keys)>1 else None,
        episodes_gained=int(((x==1)&(y==0)).sum()),episodes_lost=int(((x==0)&(y==1)).sum()))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run',type=Path);parser.add_argument('baseline',type=Path,nargs='?')
    args=parser.parse_args()
    print(json.dumps(read_summary(args.run)['groups'],indent=2))
    if args.baseline is not None:print(json.dumps(paired(args.run,args.baseline),indent=2))


if __name__=='__main__':main()
