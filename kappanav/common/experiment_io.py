"""Hash-bound experiment inputs and atomic, non-overwriting result publication."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def code_digest(*parts):
    """A short hash of the source of the given functions, classes or modules, and of any strings (constants): a
    cache keyed on it depends on exactly the code that produced it."""
    import inspect
    text = ''.join(p if isinstance(p, str) else inspect.getsource(getattr(p, 'py_func', p)) for p in parts)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def write(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        os.link(temporary, path)
    finally:
        Path(temporary).unlink()
