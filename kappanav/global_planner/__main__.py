"""python -m kappanav.global_planner {train,evaluate,infer}; data: python -m kappanav.global_planner.prepare."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from kappanav.global_planner.dataset import STORE
from kappanav.global_planner.runtime import distributed_device, digest, load_model, write_json


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='Train on the surface store (data.STORE)')
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--steps', type=int, default=40000)
    train.add_argument('--batch-size', type=int, default=8, help='512-pixel crops per rank; smaller crops fill the same pixels')
    train.add_argument('--lr', type=float, default=6e-4)
    train.add_argument('--backbone-lr', type=float, default=1e-5)
    train.add_argument('--workers', type=int, default=8)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--eval-every', type=int, default=2000)
    train.add_argument('--eval-images', type=int, default=60, help='validation images per source (0: all)')
    train.add_argument('--weight', action='append', default=[], metavar='SOURCE=W', help='override a source weight')
    train.add_argument('--resume', action='store_true', help='continue output/last.pt')
    train.add_argument('--init', type=Path, help='start from the weights of an earlier checkpoint (new schedule)')
    evaluation = commands.add_parser('evaluate', help='Per-source scores of a checkpoint')
    evaluation.add_argument('--output', type=Path, required=True)
    evaluation.add_argument('--limit', type=int, default=0, help='images per source (0: all)')
    inference = commands.add_parser('infer', help='Predict an RGB image or georeferenced TIFF')
    inference.add_argument('image', type=Path)
    inference.add_argument('--output', type=Path, required=True)
    inference.add_argument('--gsd', type=float, help='metres per pixel (read from a GeoTIFF when omitted)')
    for command in (train, evaluation):
        command.add_argument('--store', type=Path, default=STORE)
        command.add_argument('--sources', nargs='+', help='default: every supervised source in the store')
    for command in (train, evaluation, inference):
        command.add_argument('--device', default='cuda:0')
        if command is not train:
            command.add_argument('--checkpoint', type=Path, required=True)
    return top


def main():
    args = parser().parse_args()
    torch.set_num_threads(4)
    device, rank, world = distributed_device(args.device)
    if args.command == 'train':
        from kappanav.global_planner.train import train
        train(args, device, rank, world)
    elif args.command == 'evaluate':
        from kappanav.global_planner.evaluate import evaluate_checkpoint
        evaluate_checkpoint(args, device, rank, world)
    else:
        from kappanav.global_planner.inference import predict, save_prediction, working_resolution
        if world != 1:
            raise ValueError('Single-image inference runs in one process')
        if args.output.exists():
            raise FileExistsError(args.output)
        profile, gsd = None, args.gsd
        if args.image.suffix.lower() in ('.tif', '.tiff'):
            import rasterio
            with rasterio.open(args.image) as image:
                rgb, profile = image.read([1, 2, 3]).transpose(1, 2, 0), image.profile.copy()
                if gsd is None and image.crs is not None and image.crs.is_projected:
                    gsd = abs(image.transform.a)
        else:
            with Image.open(args.image) as image:
                rgb = np.asarray(image.convert('RGB'))
        if gsd is None:
            raise ValueError('Give --gsd (metres per pixel) for an image without a projected georeference')
        native = rgb.shape[:2]
        rgb, work = working_resolution(rgb, gsd)
        if profile is not None and work != gsd:
            from rasterio.transform import Affine
            t = profile['transform']
            profile.update(transform=t * Affine.scale(native[1] / rgb.shape[1], native[0] / rgb.shape[0]),
                           width=rgb.shape[1], height=rgb.shape[0])
        model, state = load_model(args.checkpoint, device)
        posterior = predict(model, rgb, work, device)
        save_prediction(args.output, rgb, work, posterior, profile)
        write_json(args.output / 'provenance.json', {'checkpoint': str(args.checkpoint.resolve()),
                   'checkpoint_sha256': state['loaded_checkpoint_sha256'], 'image_sha256': digest(args.image),
                   'classes': state['classes'], 'input_gsd_m': gsd, 'working_gsd_m': work,
                   'native_shape': list(native), 'working_shape': list(rgb.shape[:2])})
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
