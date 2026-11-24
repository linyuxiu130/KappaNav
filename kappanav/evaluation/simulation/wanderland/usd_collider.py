"""Read the collision mesh from a Wanderland scene.usdz without a USD runtime."""
import re
import zipfile

import numpy as np


def _array(text, name):
    """The text of `name = [...]`; no array here nests brackets, so it ends at the first ']'."""
    m = re.search(name + r'\s*=\s*\[', text)
    if m is None:
        raise ValueError(f'{name} not found')
    return text[m.end():text.index(']', m.end())]


def read_collider(usdz_path):
    """Return (vertices [N,3] float64, faces [M,3] int64, info): the mesh's local points (dataset frame)."""
    with zipfile.ZipFile(usdz_path) as z:
        name = next(n for n in z.namelist() if n.endswith('.usda'))
        text = z.read(name).decode('utf-8', 'replace')
    up = re.search(r'upAxis\s*=\s*"(\w)"', text)
    mpu = re.search(r'metersPerUnit\s*=\s*([\d.eE+-]+)', text)
    block = text[text.index('def Xform "Collider"'):]
    # Some releases author the stage placement on the Collider prim, others on the Visual prim; the mesh's local
    # points are always in the dataset frame shared by raw_pcd.ply and episodes.json (prepare.frame_check), so the
    # placement is not applied.
    counts = np.array(_array(block, r'int\[\] faceVertexCounts').split(','), dtype=np.int64)
    if not (counts == 3).all():
        raise ValueError('Expected a triangle mesh')
    idx = np.array(_array(block, r'int\[\] faceVertexIndices').split(','), dtype=np.int64)
    pts = _array(block, r'point3f\[\] points')
    verts = np.array(re.findall(r'[-\d.eE+]+', pts), dtype=np.float64).reshape(-1, 3)
    info = dict(up_axis=up.group(1) if up else None, meters_per_unit=float(mpu.group(1)) if mpu else 1.0)
    return verts, idx.reshape(-1, 3), info
