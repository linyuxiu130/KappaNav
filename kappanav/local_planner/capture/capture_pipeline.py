"""Expert captures -> in-memory labels and candidates -> ready-to-train scene arrays.

A worker processes whole source grids, so expert pools include every capture of a scene: it claims the largest
unclaimed scene, drives its captures concurrently (CPU planning in capture_helpers processes) and labels each
capture as it is ready. Raw captures are retained for relabelling; no per-anchor/per-row tensor files are written.
Completed scenes are independent of worker and helper counts; the final pass only publishes catalogs.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import argparse, fcntl, hashlib, json, os, queue, tempfile, threading
from pathlib import Path
import numpy as np
import torch
from kappanav.local_planner.data.capture import load_capture
from kappanav.local_planner.data.capture_helpers import Helpers, drive_plan
from kappanav.local_planner.capture.collect_mpc import collect, acquisition_contract, episode_camera
from kappanav.local_planner.data.prepare_replay import label_records, record_candidate_banks
from kappanav.local_planner.data.prepare_arrays import prepare_scene, finalize_scenes
from kappanav.local_planner.data.privileged import SourceConfigurationSpaceQuery
from kappanav.local_planner.data.value_targets import RouteUtilityTeacher
from kappanav.local_planner.inference.bundle import load_bundle
from kappanav.common.experiment_io import sha, write
from kappanav.evaluation.simulation import scene
from kappanav.local_planner.control.mpc import PathMPC


def scene_jobs(scenarios, root, bundle, runtime, pitch_up_range=None):
    specs = json.loads(Path(scenarios).read_text())['conditions']
    groups, grids = OrderedDict(), {}
    for spec in specs:
        navigation = spec['case']['navigation']
        if navigation not in grids: grids[navigation] = sha(navigation)
        groups.setdefault(grids[navigation], []).append(spec)
    code = hashlib.sha256()
    source = Path(__file__).resolve().parents[2]
    for path in sorted(source.rglob('*.py')):
        code.update(str(path.relative_to(source)).encode()); code.update(path.read_bytes())
    common = dict(format='direct_scene_arrays_v1', driver='expert', bundle_sha256=sha(bundle),
                  runtime_sha256=sha(runtime), code_sha256=code.hexdigest(),
                  mount_pitch_up_range_degrees=None if pitch_up_range is None else list(map(float, pitch_up_range)))
    jobs = []
    for grid_sha, tasks in groups.items():
        splits = {task['split'] for task in tasks}
        if len(splits) != 1 or not splits <= {'train', 'validation'}:
            raise ValueError('A source grid must belong to exactly one train/validation split')
        provenance = dict(common, source_grid_sha256=grid_sha, split=tasks[0]['split'], tasks=tasks)
        jobs.append((Path(root)/'shards'/grid_sha, tasks, provenance))
    return jobs


def read_capture(folder, spec, grid_sha, camera):
    """A reused recording must be the requested task on the same grid and camera, captured under the current
    acquisition contract (capture code, expert controller and runtime calibration are all in its hash)."""
    raw = load_capture(folder/'capture.json')
    if raw['case'] != spec['case'] or raw['provenance']['trial_id'] != spec['trial_id'] or raw.get('camera') != camera \
            or raw['acquisition'] != acquisition_contract() or sha(folder/'source_grid.npz') != grid_sha:
        raise ValueError(f'Capture {folder} was recorded for other inputs; use a new output root')
    return json.loads((folder/'collection.json').read_text()), raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=scene.DEFAULT_RUNTIME)
    parser.add_argument('--pitch-up-degrees', type=float, nargs=2, metavar=('LOW', 'HIGH'),
                        help='draw each episode\'s mount pitch-up uniformly in [LOW, HIGH] (default: the runtime camera)')
    parser.add_argument('--worker', type=int); parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--helpers', type=int, help='CPU helper processes per worker (default 2*cores/workers)')
    args = parser.parse_args()
    if args.workers < 1 or (args.worker is not None and not 0 <= args.worker < args.workers):
        parser.error('workers must be positive and worker must be in [0, workers)')
    torch.set_num_threads(1)
    args.root.mkdir(parents=True, exist_ok=True)
    jobs = scene_jobs(args.scenarios, args.root, args.bundle, args.runtime, args.pitch_up_degrees)
    pending = []
    for destination, tasks, provenance in jobs:
        marker = destination/'scene.json'
        if marker.exists():
            recorded = json.loads(marker.read_text())['provenance']
            # Indexing consumes recorded shards, whose producer revision may differ from the indexer.
            expected = dict(provenance, code_sha256=recorded['code_sha256']) if args.worker is None else provenance
            if recorded != expected:
                raise ValueError(f'Scene inputs changed: {destination}; use a new output root')
        else: pending.append((destination, tasks, provenance))
    if args.worker is None:
        if pending: raise ValueError(f'{len(pending)} scenes are unfinished; run capture workers first')
        dataset = finalize_scenes([job[0] for job in jobs], args.root)
        rows = [json.loads((args.root/'captures'/spec['case_id']/'collection.json').read_text()) for _, tasks, _ in jobs for spec in tasks]
        manifest = dict(complete=True, episodes=rows)
        path = args.root/'collections.json'
        if path.exists():
            if json.loads(path.read_text()) != manifest: raise ValueError('Collection summary changed; use a new output root')
        else: write(path, manifest)
        print(json.dumps(dict(dataset=str(dataset))), flush=True)
        return
    if not pending: return
    scene.configure(args.runtime)
    runtime = json.loads(args.runtime.read_text())
    actor = load_bundle(args.bundle, 'cuda')
    captures = args.root/'captures'; captures.mkdir(exist_ok=True)
    claims = args.root/'claims'; claims.mkdir(exist_ok=True)
    helpers = args.helpers or max(1, 2*len(os.sched_getaffinity(0))//args.workers)
    with tempfile.TemporaryDirectory(prefix='mpc_codegen_', dir=args.root) as scratch, Helpers(helpers) as pool:
        controllers = Controllers(Path(scratch)/'solver')
        # Largest scenes first; any worker takes any unclaimed scene, so the workers finish together.
        for destination, tasks, provenance in sorted(pending, key=scene_cost, reverse=True):
            with claim(claims/provenance['source_grid_sha256']) as claimed:
                if not claimed or (destination/'scene.json').exists(): continue
                metadata = run_scene(actor, destination, tasks, provenance, args, runtime, captures, pool, controllers)
                print(json.dumps(dict(scene=str(destination), rows=metadata['rows'])), flush=True)
            torch.cuda.empty_cache()


def scene_cost(job):
    """Straight-line task lengths: a proxy for a scene's anchors, to schedule large scenes first."""
    return sum(float(np.linalg.norm(np.subtract(task['case']['goal'], task['case']['start'][:2]))) for task in job[1])


@contextmanager
def claim(path):
    """An exclusive advisory lock on a scene while one worker processes it; the OS releases it if the worker dies."""
    with open(path, 'a') as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False; return
        yield True


class Controllers:
    """MPC solvers for concurrent captures: one generated solver, one instance per driving thread."""
    def __init__(self, folder):
        self.folder, self.free, self.count, self.lock = folder, queue.SimpleQueue(), 0, threading.Lock()

    @contextmanager
    def __call__(self):
        try: controller = self.free.get_nowait()
        except queue.Empty:
            with self.lock:
                controller = PathMPC(self.folder, generate=self.count == 0); self.count += 1
        try: yield controller
        finally: self.free.put(controller)


# Several label threads keep the planning helpers busy while each does a capture's serial work.
LABEL_THREADS = 4


def run_scene(actor, destination, tasks, provenance, args, runtime, captures, helpers, controllers):
    """Drive a scene's missing captures concurrently and label every capture as it is ready, then compile the scene.

    Each capture's driving thread renders through this (main) thread's simulator and plans in the helpers;
    LABEL_THREADS label threads label finished captures (anchor geometry in the helpers) and record their candidate
    banks. A capture's labels depend only on its own files (its seed, history and fixed candidate noise), so they
    do not depend on which thread labels it; the records are compiled in task order, as by one sequential worker."""
    torch.manual_seed(actor.config.training.seed + int(provenance['source_grid_sha256'][:8], 16))
    utility = RouteUtilityTeacher(SourceConfigurationSpaceQuery.from_paths((Path(tasks[0]['case']['navigation']),)),
                                  actor.policy.planning_horizon_m, actor.head.radius)
    ready, labelled, failures = queue.SimpleQueue(), {}, []

    def label():
        while (item := ready.get()) is not None:
            index, folder, row, raw = item
            try:
                records = label_records(actor, folder, raw=raw, utility=utility, helpers=helpers)
                record_candidate_banks(actor, records)
            except BaseException as error:
                failures.append(error); return
            labelled[index] = records
            print(json.dumps(row), flush=True)

    labellers = [threading.Thread(target=label, daemon=True) for _ in range(LABEL_THREADS)]
    for labeller in labellers: labeller.start()
    groups = OrderedDict()
    try:
        for index, spec in enumerate(tasks):
            folder = captures/spec['case_id']
            if (folder/'collection.json').is_file():
                camera, _ = episode_camera(spec, runtime['camera'], args.pitch_up_degrees)
                ready.put((index, folder, *read_capture(folder, spec, provenance["source_grid_sha256"], camera)))
                continue
            if folder.exists(): raise ValueError(f'Incomplete capture at {folder}; preserve it and use a new output root')
            case = spec['case']
            groups.setdefault((case['assets'], case['asset_scene_id'], case['scene_dataset']), []).append(index)
        for key, indices in groups.items():
            drive_captures(key, [(index, tasks[index]) for index in indices], runtime, captures, helpers, controllers, ready,
                           args.pitch_up_degrees)
    finally:
        for labeller in labellers: ready.put(None)
        for labeller in labellers: labeller.join()
    if failures: raise failures[0]
    records = [record for index in range(len(tasks)) for record in labelled[index]]
    return prepare_scene(actor, records, destination, provenance=provenance)


def drive_captures(key, specs, runtime, captures, helpers, controllers, ready, pitch_up_range=None):
    """Expert-drive captures of one simulator scene concurrently; the simulator renders only in this thread."""
    assets, scene_id, dataset = key
    simulator = scene.create_simulator(Path(assets), scene_id, 0, runtime['camera'], dataset)
    try:
        _, k, _ = scene.sensor_assets()
        requests = queue.SimpleQueue()

        def render(camera, view):
            reply = queue.SimpleQueue(); requests.put((view, camera, reply)); depth = reply.get()
            if isinstance(depth, BaseException): raise depth
            return depth

        def drive(index, spec):
            folder = captures/spec['case_id']; case = spec['case']
            camera, _ = episode_camera(spec, runtime['camera'], pitch_up_range)
            observe = lambda view: render(camera, view)
            goal = np.asarray(case['goal'], dtype=np.float64)
            plan = lambda view: helpers.drive(drive_plan, case['navigation'], view, goal)
            helpers.drivers(1)
            try:
                with controllers() as controller, tempfile.TemporaryDirectory(prefix='.capture-', dir=captures) as temporary:
                    prepared = Path(temporary)/'capture'
                    row, raw = collect(spec, prepared, controller, simulator=None, observe=observe, plan=plan, camera=camera)
                    os.rename(prepared, folder)
            finally: helpers.drivers(-1)
            ready.put((index, folder, row, raw))

        with ThreadPoolExecutor(len(specs)) as threads:
            futures = [threads.submit(drive, index, spec) for index, spec in specs]
            while not all(future.done() for future in futures):
                try: view, camera, reply = requests.get(timeout=.005)
                except queue.Empty: continue
                try: reply.put(scene.observation(simulator, view, k, camera))
                except BaseException as error: reply.put(error)
            for future in futures: future.result()
    finally:
        simulator.close()


if __name__ == '__main__': main()
