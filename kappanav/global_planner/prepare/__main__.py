"""python -m kappanav.global_planner.prepare SOURCE [SOURCE ...] [--workers N]: convert sources into the store."""
import argparse
import json

from kappanav.global_planner.prepare import STORE, write_source
from kappanav.global_planner.prepare.local import SOURCES as LOCAL
from kappanav.global_planner.prepare.downloaded import SOURCES as DOWNLOADED

SOURCES = {**LOCAL, **DOWNLOADED}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', choices=sorted(SOURCES))
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    for name in args.sources:
        meta, items = SOURCES[name]()
        meta = write_source(name, meta, items, args.workers)
        print(json.dumps({k: meta[k] for k in ('source', 'images', 'splits', 'scene_groups')}), flush=True)
    print(f'store: {STORE}')


if __name__ == '__main__':
    main()
