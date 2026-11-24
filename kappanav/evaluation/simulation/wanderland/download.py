"""Mirror the navigation-relevant files of every Wanderland v1 scene from Hugging Face.

Skips the raw image archives (images/fisheye .tar.gz), which only serve novel-view synthesis.
Each file is fetched to a .part file, checked against the Hub's size and LFS sha256,
then renamed; finished files are skipped on rerun.
"""
import hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(os.environ.get('KAPPANAV_WANDERLAND', 'data/wanderland'))
RAW = ROOT / 'raw'
BASE = 'https://huggingface.co/datasets/ai4ce/wanderland/resolve/main/'
SKIP = ('images.tar.gz', 'fisheye.tar.gz')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def fetch(entry):
    path, size = entry['path'], entry['size']
    oid = (entry.get('lfs') or {}).get('oid')
    dest = RAW / path
    if dest.exists() and dest.stat().st_size == size:
        return path, 'skip'
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    for attempt in range(5):
        r = subprocess.run(['curl', '-sSL', '--fail', '-C', '-', '--retry', '5', '--retry-delay', '10',
                            '-o', str(part), BASE + path], capture_output=True, text=True)
        if r.returncode == 0 and part.exists() and part.stat().st_size == size:
            break
        if part.exists() and part.stat().st_size > size:
            part.unlink()
        time.sleep(15 * (attempt + 1))
    else:
        return path, f'FAILED curl={r.returncode} {r.stderr.strip()[:200]}'
    if oid and sha256(part) != oid:
        part.unlink()
        return path, 'FAILED sha256 mismatch'
    part.rename(dest)
    return path, 'ok'


def main(workers=12):
    files = json.load(open(ROOT / 'hf_files.json'))
    todo = [f for f in files if f['path'].startswith('data/v1/') and not f['path'].endswith(SKIP)]
    # Files the scene pipeline needs come first (scene by scene), the 3DGS models last.
    rank = {'3dgs.ply': 1}
    todo.sort(key=lambda f: (rank.get(f['path'].rsplit('/', 1)[-1], 0), f['path'].rsplit('/', 1)[0], f['size']))
    total = sum(f['size'] for f in todo)
    print(f'{len(todo)} files, {total/1e9:.1f} GB', flush=True)
    done = failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(fetch, f) for f in todo]):
            path, status = fut.result()
            done += 1
            if status.startswith('FAILED'):
                failed += 1
                print(path, status, flush=True)
            if done % 100 == 0 or done == len(todo):
                print(f'progress {done}/{len(todo)} failed={failed}', flush=True)
    print('DONE' if failed == 0 else f'DONE with {failed} failures', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
