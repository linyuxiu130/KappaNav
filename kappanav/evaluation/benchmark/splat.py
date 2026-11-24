"""Camera images of a Wanderland scene from the dataset's 3D Gaussian splatting (3dgs.ply), at canonical camera poses.

Wanderland's Habitat mesh is rebuilt from LiDAR and carries no colour; the released 3DGS was trained on the capture's
ground-level images, so it renders what the robot's camera would see. It runs in its own environment (gsplat,
KAPPANAV_GSPLAT_PYTHON), so evaluation.video calls it as a process:

    python -m kappanav.evaluation.benchmark.splat SCENE.ply POSES.npz OUT.npz

POSES.npz: camera_to_world [N, 4, 4] (optical camera to canonical world: x right, y down, z forward), K [3, 3], width,
height. OUT.npz: rgb [N, H, W, 3] uint8 and depth [N, H, W] (camera z, inf where the splats are transparent).
The 3DGS lives in the dataset frame (the capture's COLMAP frame, +Y down), canonical = (x, z, -y) of it.
"""
import sys

import numpy as np

PLY_TYPES = {'float': '<f4', 'double': '<f8', 'uchar': 'u1', 'int': '<i4', 'uint': '<u4'}
CANONICAL_TO_DATASET = np.array([[1., 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
OPAQUE = .5  # a pixel whose accumulated opacity is below this shows no surface (depth inf)


def read_ply(path):
    """The vertex table of a binary little-endian PLY as a numpy structured array."""
    with open(path, 'rb') as f:
        fields, count = [], 0
        while (line := f.readline().decode().strip()) != 'end_header':
            if line.startswith('element vertex'):
                count = int(line.split()[-1])
            elif line.startswith('property'):
                _, kind, name = line.split()
                fields.append((name, PLY_TYPES[kind]))
        return np.fromfile(f, dtype=np.dtype(fields), count=count)


def load_splats(path, device):
    import torch
    v = read_ply(path)
    names = v.dtype.names
    rest = sorted((n for n in names if n.startswith('f_rest_')), key=lambda n: int(n[7:]))
    column = lambda keys: torch.from_numpy(np.stack([v[k] for k in keys], 1).astype(np.float32)).to(device)
    sh = torch.cat([column(['f_dc_0', 'f_dc_1', 'f_dc_2'])[:, None],
                    column(rest).reshape(len(v), 3, -1).transpose(1, 2)], 1)  # N x (degree + 1)^2 x 3
    return dict(means=column(['x', 'y', 'z']), quats=column(['rot_0', 'rot_1', 'rot_2', 'rot_3']),
                scales=column(['scale_0', 'scale_1', 'scale_2']).exp(),
                opacities=column(['opacity'])[:, 0].sigmoid(), colors=sh,
                sh_degree=int(round(np.sqrt(sh.shape[1]))) - 1)


def render(splats, camera_to_world, K, width, height, batch=8):
    import torch
    from gsplat import rasterization
    device = splats['means'].device
    view = np.linalg.inv(CANONICAL_TO_DATASET @ np.asarray(camera_to_world, np.float64))  # dataset world -> camera
    rgb, depth = [], []
    for k in range(0, len(view), batch):
        viewmats = torch.as_tensor(view[k:k + batch], dtype=torch.float32, device=device)
        Ks = torch.as_tensor(np.repeat(np.asarray(K, np.float32)[None], len(viewmats), 0), device=device)
        with torch.no_grad():
            image, alpha, _ = rasterization(viewmats=viewmats, Ks=Ks, width=width, height=height,
                                            render_mode='RGB+ED', **splats)
        d = image[..., 3]
        d[alpha[..., 0] < OPAQUE] = float('inf')
        rgb.append((image[..., :3].clamp(0, 1) * 255).round().byte().cpu().numpy())
        depth.append(d.cpu().numpy())
    return np.concatenate(rgb), np.concatenate(depth)


def main():
    import torch
    ply, poses, out = sys.argv[1:4]
    with np.load(poses) as p:
        camera_to_world, K, width, height = p['camera_to_world'], p['K'], int(p['width']), int(p['height'])
    rgb, depth = render(load_splats(ply, torch.device('cuda')), camera_to_world, K, width, height)
    np.savez(out, rgb=rgb, depth=depth)


if __name__ == '__main__':
    main()
