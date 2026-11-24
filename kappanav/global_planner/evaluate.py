"""Whole-image evaluation per source on its test rows (data.Source.test_ids), each at WORK_GSD_M."""
import json
import time

import numpy as np
import torch
import torch.distributed as dist

from kappanav.global_planner.dataset import Source, evaluation_image
from kappanav.global_planner.inference import predict
from kappanav.global_planner.metrics import Scores
from kappanav.global_planner.runtime import load_model, write_json
from kappanav.global_planner.taxonomy import drivable_probability, traversability


def evaluation_sets(store, names, limit=0, role='val'):
    """{source: rows} of the labelled evaluation rows; `limit` > 0 keeps a fixed subset per source. The default is the
    fixed test split (reported scores); training passes role='select' so the model is never chosen on the test rows."""
    sets = {}
    for name in names:
        source = Source(name, store)
        rows = source.split(role)
        if limit and len(rows) > limit:
            rows = [rows[i] for i in np.random.default_rng(0).choice(len(rows), limit, replace=False)]
        if rows:
            sets[name] = (source, rows)
    return sets


@torch.inference_mode()
def evaluate(model, sets, device, rank=0, world=1):
    model.eval()
    started, report = time.perf_counter(), {}
    for name, (source, rows) in sets.items():
        scores = Scores()
        for row in rows[rank::world]:
            rgb, labels, gsd = evaluation_image(source, row)
            posterior = predict(model, rgb, gsd, device)
            scores.add(posterior, drivable_probability(posterior), traversability(posterior), labels)
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, scores)
            scores = gathered[0]
            for other in gathered[1:]:
                scores.merge(other)
        report[name] = {**scores.report(), 'images': len(rows), 'scene_disjoint': source.scene_disjoint}
    mean = lambda key: (lambda v: float(np.mean(v)) if v else None)([r[key] for r in report.values() if r[key] is not None])
    return {'sources': report, **{f'mean_{k}': mean(k) for k in ('value_error', 'passable_AUC', 'false_passable', 'missed_passable', 'mIoU')},
            'seconds': time.perf_counter() - started}


def evaluate_checkpoint(args, device, rank, world):
    if args.output.exists():
        raise FileExistsError(args.output)
    model, state = load_model(args.checkpoint, device)
    sets = evaluation_sets(args.store, args.sources or state['config']['sources'], args.limit)
    result = evaluate(model, sets, device, rank, world)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, {**result, 'checkpoint': str(args.checkpoint.resolve()),
                                'checkpoint_sha256': state['loaded_checkpoint_sha256'], 'step': state['step']})
        print(json.dumps(result), flush=True)
