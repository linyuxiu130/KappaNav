"""Which part of the package may import which: the layering of README.md."""
import ast
from pathlib import Path

import kappanav

PACKAGE = Path(kappanav.__file__).parent
ALLOWED = {'common': set(), 'global_planner': {'common'}, 'local_planner': {'common'},
           'evaluation': {'common', 'global_planner', 'local_planner'}}
LOCAL = ('model', 'features', 'control', 'inference', 'data', 'training', 'capture')   # each may import the earlier ones


def modules():
    for path in PACKAGE.rglob('*.py'):
        yield path.relative_to(PACKAGE).with_suffix('').parts, path


def imports(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
            names += [f'{node.module}.{a.name}' for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        yield from (n.split('.')[1:] for n in names if n.startswith('kappanav.'))


def test_the_parts_only_import_the_parts_below_them():
    wrong = []
    for parts, path in modules():
        if len(parts) < 2:
            continue
        allowed = ALLOWED[parts[0]] | {parts[0]} | ({'evaluation'} if parts[:2] == ('local_planner', 'capture') else set())
        wrong += [(*parts, '->', *name) for name in imports(path) if name[0] not in allowed]
    assert not wrong, wrong


def test_the_local_planner_is_layered():
    wrong = []
    for parts, path in modules():
        if parts[0] == 'local_planner' and len(parts) > 2:
            below = LOCAL[:LOCAL.index(parts[1]) + 1]
            wrong += [(*parts, '->', *n) for n in imports(path) if n[0] == 'local_planner' and len(n) > 1 and n[1] in LOCAL
                      and n[1] not in below]
    assert not wrong, wrong
