"""Resolve official scene geometry into a depth-renderable mesh, and its appearance into a colour-renderable one."""
from pathlib import Path
import hashlib,io,json,os
import numpy as np


def habitat_from_usd(up_axis: str, meters_per_unit: float) -> np.ndarray:
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("USD stage units must be finite and positive")
    rotation = {"Y": np.eye(3), "Z": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])}[up_axis]
    result = np.eye(4)
    result[:3, :3] = rotation * meters_per_unit
    return result

# Base-colour textures are downscaled to at most this, except on the ground (`ground`), which a camera 0.25 m above it
# sees at millimetres per pixel: a city's ground is a few baked 4096 px tiles of about 0.1 km each (3 cm per texel).
TEXTURE_MAX_PX = 1024
GROUND_TEXTURE_MAX_PX = 4096
TEXTURE_JPEG_QUALITY = 90  # textures are stored as JPEG: as PNG a city's asset outgrows glTF's 4 GB binary limit
DEFAULT_BASE_COLOR = (.18, .18, .18)  # a surface whose material or texture cannot be resolved


def export_geometry(source: Path, search_root: Path, destination: Path, *, root_path="/Root/Meshes", fix_blankets=True,
                    appearance: Path | None = None, ground: str | None = None) -> dict:
    """The scene's visible meshes as one glTF (the depth-render and collision asset). With `appearance`, also the
    colour-render asset there: the same triangles with their UsdPreviewSurface base colour (constant, or the
    texture with the mesh's texture coordinates) per material binding and material subset. Meshes whose prim path
    contains `ground` keep textures up to GROUND_TEXTURE_MAX_PX."""
    from pxr import Ar, Gf, Sdf, Usd, UsdGeom
    import trimesh

    context = Ar.ResolverContext(Ar.DefaultResolverContext([str(search_root)]))
    stage = Usd.Stage.Open(str(source), context)
    # A partially composed stage can still render plausible but unsafe depths.
    errors = [str(error) for p in stage.Traverse() for error in p.GetPrimIndex().localErrors]
    if errors:
        raise ValueError("unresolved USD composition: " + "\n".join(errors))
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError(f"missing geometry root {root_path}: {source}")
    # Match the reference scene loader's blanket transform before exporting either geometry.
    stage.SetEditTarget(stage.GetSessionLayer())
    for prim in Usd.PrimRange(root):
        if fix_blankets and "blanket" in prim.GetName().lower():
            for child in prim.GetAllChildren():
                scale=child.GetAttribute("xformOp:scale")
                if not scale:scale=child.CreateAttribute("xformOp:scale",Sdf.ValueTypeNames.Double3,False)
                scale.Set(Gf.Vec3d(.01,.01,.01))
                order=child.GetAttribute("xformOpOrder")
                if not order:order=child.CreateAttribute("xformOpOrder",Sdf.ValueTypeNames.TokenArray,False)
                operations=list(order.Get() or [])
                if "xformOp:scale" not in operations:order.Set(operations+["xformOp:scale"])
    units = UsdGeom.GetStageMetersPerUnit(stage)
    axis = str(UsdGeom.GetStageUpAxis(stage))
    basis = habitat_from_usd(axis, units)
    transforms = UsdGeom.XformCache()
    result = trimesh.Scene()
    triangles = 0
    material = trimesh.visual.material.PBRMaterial(doubleSided=True)
    painter = _Appearance(ground) if appearance is not None else None
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible or mesh.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        authored_counts = mesh.GetFaceVertexCountsAttr().Get()
        if not authored_counts:
            # GRScenes authors Mesh containers with geometry on child prims.
            continue
        counts = np.asarray(authored_counts, dtype=np.int64)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if (counts < 3).any() or indices.size != counts.sum():
            raise ValueError(f"invalid polygon geometry: {prim.GetPath()}")
        if not np.isfinite(points).all() or indices.min() < 0 or indices.max() >= len(points):
            raise ValueError(f"invalid USD mesh: {prim.GetPath()}")
        corners, polygons = triangle_corners(counts, mesh.GetHoleIndicesAttr().Get() or [])
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            corners = corners[:, ::-1]
        # Gf matrices act on row vectors; trimesh/glTF matrices on columns.
        world = basis @ np.asarray(transforms.GetLocalToWorldTransform(prim)).T
        # Habitat/Recast does not repair winding for reflected glTF nodes.
        # A = (A F) F, F=diag(-1,1,1): bake F into vertices and reverse
        # winding, leaving a positive-determinant instance with identical points.
        if np.linalg.det(world[:3, :3]) < 0:
            points = points * [-1, 1, 1]
            corners = corners[:, ::-1]
            world = world @ np.diag([-1., 1., 1., 1.])
        faces = indices[corners]
        if painter is not None:
            painter.add(prim, mesh, points, corners, polygons, indices, world)
        digest = hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest()
        if digest not in result.geometry:
            geometry = trimesh.Trimesh(points, faces, process=False)
            geometry.visual = trimesh.visual.TextureVisuals(material=material)
            result.add_geometry(geometry, geom_name=digest,
                                node_name=str(prim.GetPath()), transform=world)
        else:
            result.graph.update(frame_to=str(prim.GetPath()), matrix=world, geometry=digest)
        triangles += len(faces)
    if not triangles:
        raise ValueError(f"empty scene: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".glb.partial")
    temporary.write_bytes(result.export(file_type="glb", include_normals=False))
    os.replace(temporary, destination)
    meta = {"triangles": triangles, "meshes": len(result.geometry),
            "usd_up_axis": axis, "usd_meters_per_unit": units,
            "habitat_from_usd": basis.tolist(), "bounds_m": result.bounds.tolist(),
            "size_bytes": destination.stat().st_size}
    if painter is not None:
        temporary = appearance.with_suffix(".glb.partial")
        temporary.write_bytes(painter.scene.export(file_type="glb"))
        os.replace(temporary, appearance)
        meta["appearance"] = dict(painter.stats, size_bytes=appearance.stat().st_size, texture_max_px=TEXTURE_MAX_PX,
                                  ground_texture_max_px=GROUND_TEXTURE_MAX_PX)
    return meta


class _Appearance:
    """The colour-render scene: per mesh and material, unwelded triangles (texture coordinates are per face corner)
    with the material's base colour."""
    def __init__(self, ground=None):
        import trimesh
        self.ground, self.scene, self.materials, self.images = ground, trimesh.Scene(), {}, {}
        self.stats = dict(triangles=0, textured_triangles=0, missing_textures=[], unresolved_materials=0)

    def add(self, prim, mesh, points, corners, polygons, indices, world):
        import trimesh
        from pxr import UsdGeom, UsdShade
        bound = {}  # polygon -> material prim
        for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            target = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            if target:
                for polygon in subset.GetIndicesAttr().Get() or []:
                    bound[int(polygon)] = target
        default = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        keys = [bound.get(int(p), default) for p in polygons]
        paths = np.array([str(k.GetPath()) if k else '' for k in keys])
        for path in np.unique(paths):
            chosen = paths == path
            target = keys[int(np.flatnonzero(chosen)[0])]
            material, varname = self.material(target, GROUND_TEXTURE_MAX_PX if self.ground and self.ground in str(prim.GetPath())
                                              else TEXTURE_MAX_PX)
            corner = corners[chosen]
            uv = np.zeros((corner.size, 2))
            if varname:
                primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar(varname)
                values = primvar.ComputeFlattened() if primvar else None
                if values is not None and len(values):
                    values = np.asarray(values, np.float64)
                    interpolation = primvar.GetInterpolation()
                    lookup = {UsdGeom.Tokens.faceVarying: corner, UsdGeom.Tokens.vertex: indices[corner],
                              UsdGeom.Tokens.varying: indices[corner],
                              UsdGeom.Tokens.uniform: np.repeat(polygons[chosen][:, None], 3, 1)}.get(interpolation)
                    if lookup is not None and lookup.max() < len(values):
                        uv = values[lookup].reshape(-1, 2)
                        self.stats['textured_triangles'] += len(corner)
            vertices = points[indices[corner]].reshape(-1, 3)
            geometry = trimesh.Trimesh(vertices, np.arange(len(vertices)).reshape(-1, 3), process=False,
                                       visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
            name = hashlib.sha256(vertices.tobytes() + uv.tobytes() + path.encode()).hexdigest()
            if name not in self.scene.geometry:
                self.scene.add_geometry(geometry, geom_name=name, node_name=f'{prim.GetPath()}:{path}', transform=world)
            else:
                self.scene.graph.update(frame_to=f'{prim.GetPath()}:{path}', matrix=world, geometry=name)
            self.stats['triangles'] += len(corner)

    def material(self, target, max_px):
        """(trimesh material, texture-coordinate primvar name or None) of a USD material prim, its texture at most
        max_px."""
        import trimesh
        from pxr import UsdShade
        key = (str(target.GetPath()) if target else '', max_px)
        if key in self.materials:
            return self.materials[key]
        color, image, varname = DEFAULT_BASE_COLOR, None, None
        shader = target.ComputeSurfaceSource()[0] if target else None
        if shader and shader.GetIdAttr().Get() == 'UsdPreviewSurface':
            diffuse = shader.GetInput('diffuseColor')
            source = diffuse.GetConnectedSource() if diffuse else None
            if source:
                texture = UsdShade.Shader(source[0].GetPrim())
                file = texture.GetInput('file').Get() if texture.GetInput('file') else None
                scale = texture.GetInput('scale').Get() if texture.GetInput('scale') else None
                color = (1., 1., 1.) if scale is None else tuple(float(v) for v in scale)[:3]
                image = self.image(file, max_px)
                st = texture.GetInput('st').GetConnectedSource() if texture.GetInput('st') else None
                reader = UsdShade.Shader(st[0].GetPrim()) if st else None
                varname = reader.GetInput('varname').Get() if reader and reader.GetInput('varname') else 'st'
                if image is None:
                    color = DEFAULT_BASE_COLOR
            elif diffuse and diffuse.Get() is not None:
                color = tuple(float(v) for v in diffuse.Get())
        else:
            self.stats['unresolved_materials'] += 1
        material = trimesh.visual.material.PBRMaterial(baseColorFactor=[*color, 1.], baseColorTexture=image,
                                                       metallicFactor=0., roughnessFactor=1., doubleSided=True)
        self.materials[key] = material, (varname if image is not None else None)
        return self.materials[key]

    def image(self, file, max_px):
        """The texture as an RGB image no larger than max_px, or None (recorded) when it cannot be read."""
        from PIL import Image
        path = getattr(file, 'resolvedPath', '') if file is not None else ''
        if (path, max_px) not in self.images:
            try:
                image = Image.open(path).convert('RGB')
                image.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
                encoded = io.BytesIO()
                image.save(encoded, 'JPEG', quality=TEXTURE_JPEG_QUALITY)
                self.images[path, max_px] = Image.open(encoded)  # a JPEG image: the glTF export keeps the format
            except (OSError, ValueError):
                self.stats['missing_textures'].append(getattr(file, 'path', str(file)))
                self.images[path, max_px] = None
        return self.images[path, max_px]


def triangle_corners(counts, holes=()):
    """USD's fan triangulation of authored polygons, holes dropped, winding kept: each triangle's corners as
    positions in the face-vertex arrays, and the polygon each triangle comes from."""
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    fan = counts - 2
    polygon = np.repeat(np.arange(len(counts)), fan)
    k = np.arange(fan.sum()) - np.repeat(np.cumsum(fan) - fan, fan) + 1
    corners = np.stack([starts[polygon], starts[polygon] + k, starts[polygon] + k + 1], 1)
    keep = np.isin(polygon, np.asarray(list(holes), dtype=np.int64), invert=True)
    return corners[keep], polygon[keep]


def triangulate_faces(points, indices, counts, holes=()):
    """The triangles (vertex indices) of triangle_corners."""
    return np.asarray(indices, dtype=np.int64)[triangle_corners(counts, holes)[0]].reshape(-1, 3)


def write_stage(assets: Path, scene_id: str) -> None:
    """Register assets/stages/<scene_id>.glb (Y-up, metres) as a Habitat stage and scene of the dataset `assets`;
    rendered from assets/stages/<scene_id>.appearance.glb (export_geometry's colour asset, same triangles) when
    there is one."""
    (assets / "scenes").mkdir(parents=True, exist_ok=True)
    render = f"{scene_id}.appearance.glb" if (assets / "stages" / f"{scene_id}.appearance.glb").exists() else f"{scene_id}.glb"
    (assets / "stages" / f"{scene_id}.stage_config.json").write_text(json.dumps(
        {"render_asset": render, "collision_asset": f"{scene_id}.glb", "up": [0, 1, 0],
         "front": [0, 0, -1], "units_to_meters": 1.0}))
    (assets / "scenes" / f"{scene_id}.scene_instance.json").write_text(json.dumps(
        {"stage_instance": {"template_name": scene_id}, "object_instances": []}))
    dataset = assets / SCENE_DATASET
    if not dataset.exists():  # the same for every scene; parallel scene workers must not rewrite it under a reader
        dataset.write_text(json.dumps(
            {"stages": {"paths": {".json": ["stages"]}}, "scene_instances": {"paths": {".json": ["scenes"]}}}))


SCENE_DATASET = "scenes.scene_dataset_config.json"
