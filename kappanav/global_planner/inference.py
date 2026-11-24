"""One sliding posterior for evaluation and maps: an image at its ground resolution -> class posterior
and its graded traversability (taxonomy.traversability).

An image finer than GSD_RANGE_M is resampled to WORK_GSD_M and a coarser one to its coarse end (the model never saw
ground outside that range); within the range it
is used as it is. A side up to TILE is one window padded by reflection to a multiple of the patch size (training
crops have such extents); larger images are tiled with a positive Hann blend.
"""
import cv2
import numpy as np
import torch
from PIL import Image

from kappanav.global_planner.dataset import GSD_RANGE_M, WORK_GSD_M, normalize_uint8_batch
from kappanav.global_planner.taxonomy import CLASSES, COLORS, route_cost, traversability

TILE, STRIDE, PATCH = 512, 384, 16


def semantic_probabilities(logits):
    """Class posterior K x H x W from logits; a non-finite logit is a real fault, not something to average over."""
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite semantic logits')
    return logits.float().softmax(-3)


def tile_starts(length):
    if length <= TILE:
        return [0]
    return sorted(set(range(0, length - TILE + 1, STRIDE)) | {length - TILE})


def working_resolution(rgb, gsd):
    """The image at the resolution the model reads it, and that resolution."""
    low, high = GSD_RANGE_M
    if low <= gsd <= high:
        return rgb, gsd
    target = WORK_GSD_M if gsd < low else high
    scale = gsd / target
    size = (max(PATCH, round(rgb.shape[1] * scale)), max(PATCH, round(rgb.shape[0] * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC), target


@torch.inference_mode()
def predict(model, rgb, gsd, device, batch_size=8):
    """uint8 RGB HxWx3 at gsd m/px (already at a working resolution: see working_resolution) -> class posterior
    K x H x W."""
    height, width = rgb.shape[:2]
    if height <= TILE and width <= TILE:
        ph, pw = -height % PATCH, -width % PATCH
        image = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode='reflect' if min(height, width) > 1 else 'edge')
        batch = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(normalize_uint8_batch(batch), torch.tensor([gsd], device=device))
        return semantic_probabilities(logits)[0, :, :height, :width].cpu().numpy()
    ramp = np.maximum(np.hanning(TILE), .05).astype(np.float32)
    weight = ramp[:, None] * ramp[None, :]
    total = np.zeros((len(CLASSES), height, width), np.float32)
    norm = np.zeros((height, width), np.float32)
    windows = [(x, y) for y in tile_starts(height) for x in tile_starts(width)]
    for first in range(0, len(windows), batch_size):
        chunk = windows[first:first + batch_size]
        patches = [np.pad(rgb[y:y + TILE, x:x + TILE], ((0, max(0, TILE - min(TILE, height - y))),
                                                          (0, max(0, TILE - min(TILE, width - x))), (0, 0)), mode='reflect')
                   for x, y in chunk]
        batch = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(normalize_uint8_batch(batch), torch.full((len(chunk),), gsd, device=device))
        values = semantic_probabilities(logits).cpu().numpy()
        for value, (x, y) in zip(values, chunk):
            h, w = min(TILE, height - y), min(TILE, width - x)
            total[:, y:y + h, x:x + w] += value[:, :h, :w] * weight[:h, :w]
            norm[y:y + h, x:x + w] += weight[:h, :w]
    return total / norm[None]


def save_prediction(output, rgb, gsd, posterior, geotiff=None):
    """Semantic ids/colours/overlay, the 0-255 traversability (grey PNG and heat map) and posterior.npz, and the value
    and route cost as GeoTIFF layers when the input was georeferenced (geotiff: rasterio profile)."""
    output.mkdir(parents=True, exist_ok=False)
    labels = posterior.argmax(0).astype(np.uint8)
    Image.fromarray(labels).save(output / 'semantic-ids.png')
    Image.fromarray(COLORS[labels]).save(output / 'semantic.png')
    Image.fromarray(np.rint(rgb * .55 + COLORS[labels] * .45).astype(np.uint8)).save(output / 'overlay.png')
    value = traversability(posterior)
    cost = route_cost(value)
    grey = np.rint(np.clip(value, 0, 255)).astype(np.uint8)
    Image.fromarray(grey).save(output / 'traversability.png')
    Image.fromarray(cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)[..., ::-1]).save(output / 'traversability-heat.png')
    np.savez_compressed(output / 'posterior.npz', posterior=posterior.astype(np.float16), traversability=value,
                        route_cost=cost, gsd_m=gsd, classes=np.array(CLASSES))
    if geotiff is not None:
        import rasterio
        profile_tif = dict(geotiff, driver='GTiff', count=1, dtype='float32', nodata=None, compress='deflate')
        profile_tif.pop('photometric', None)
        for name, values in {'traversability': value, 'route_cost': cost}.items():
            with rasterio.open(output / f'{name}.tif', 'w', **profile_tif) as destination:
                destination.write(values.astype(np.float32), 1)
