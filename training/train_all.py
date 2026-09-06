#!/usr/bin/env python3
"""Batch driver: train one (material, experiment) pair per invocation.

``--data_idx`` selects the material, ``--exp_idx`` the hyper-parameter set,
``--preset`` the resolution. Captures are read from ``data/`` unless
``--data_root`` says otherwise.

    python training/train_all.py --list                    # material/experiment tables
    python training/train_all.py --data_idx 0 --dry_run    # resolved arguments
    python training/train_all.py --data_idx 0              # train
    python training/train_all.py --data_idx 0 --preset 2k  # fit the whole capture
    for i in $(seq 0 16); do python training/train_all.py --data_idx $i; done
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]  # repo root (for neuralbtf) and this folder

from neuralbtf.data import capture_shape  # noqa: E402
from neuralbtf.materials import (MATERIALS, TEST_TEMPLATE, TRAIN_TEMPLATE,  # noqa: E402
                                 crop_for, full_capture_settings)
from train import main as train_main  # noqa: E402

# Experiment 0 is the published configuration; the rest trade offset-texture
# capacity against offset-MLP width.
EXPERIMENTS = [
    {"off_texture_channels": 8, "off_neurons": 512},
    {"off_texture_channels": 16, "off_neurons": 128},
    {"off_texture_channels": 24, "off_neurons": 64},
    {"off_texture_channels": 24, "off_neurons": 128},
    {"off_texture_channels": 32, "off_neurons": 64},
    {"off_texture_channels": 32, "off_neurons": 128},
]

COMMON = {
    "n_steps": 20000,
    "batch_size": 10,
    "val_gap": 100,
    "train_len": 2000,
    "scale": 1.0,
    "loss": 0,
    "lr_str": "1e-3",
    "texture_res": 512,
    "min_res": 128,
    "texture_channels": 8,
    "off_texture_res": 32,
    "off_hidden": 1,
    "n_neurons": 512,
    "n_hidden_layers": 3,
    "xnum": 512,
    "ynum": 512,
}

# `--preset 2k` fits the whole capture instead of one tile of it: the crop and the
# texture pyramid come from the capture's own dimensions (512 -> 2048 for a 2048²
# capture, 350 -> 1400 for a 1400² one), one image per step, and a longer schedule.
# Everything else -- model, loss, split, uv sampling -- is unchanged.
#
# Validation is far more expensive here than on a tile: it renders every held-out
# direction over the whole capture, ~9 s for 46 directions of 1400², against 17 ms
# per training step. Validating every 100 steps (the tile setting, and what the
# reference driver asked for) would leave 85% of the wall clock in validation and
# stretch 150k steps to more than four hours, so the curve is sampled every 5000
# steps instead. The final val/test numbers are computed in full either way.
PRESET_2K = {
    "batch_size": 1,
    "n_steps": 150000,
    "val_gap": 5000,
    "ckpt_gap": 5000,
}
PRESETS = ("512", "2k")


def resolve_preset(args, material) -> tuple[dict, str]:
    """Crop, resolutions and schedule for the chosen ``--preset``, plus a run suffix."""
    data_path = os.path.join(args.data_root, args.train_template.format(name=material.name))

    if args.preset == "512":
        crop_index = material.crop_index if args.crop_index is None else args.crop_index
        crop = crop_for(material.name, args.crop_size, crop_index)
        return ({"crop": crop}, f"_crop_{crop_index}_loss_{COMMON['loss']}")

    if args.crop_size or args.crop_index is not None:
        raise SystemExit("--preset 2k fits the whole capture; drop --crop_size / --crop_index")

    if args.capture_size:
        width = height = args.capture_size
    elif os.path.exists(data_path):
        _, height, width = capture_shape(data_path)
    else:
        raise SystemExit(f"--preset 2k reads the capture's size from {data_path}, which is "
                         "missing; pass --capture_size to override")

    # The original 2k runs are named without the crop/loss suffix.
    return ({**full_capture_settings(width, height), **PRESET_2K}, "")


def build_argv(args, extra: list[str] | None = None) -> list[str]:
    material = MATERIALS[args.data_idx]
    experiment = EXPERIMENTS[args.exp_idx]

    preset, suffix = resolve_preset(args, material)
    crop = preset.pop("crop")

    resolutions = {**COMMON, **preset}
    ending = (f"{resolutions['n_neurons']}_{resolutions['n_hidden_layers']}_sph"
              f"_{resolutions['off_texture_res']}_{experiment['off_texture_channels']}"
              f"_{resolutions['off_hidden']}_{experiment['off_neurons']}{suffix}")

    settings = {
        **resolutions,
        **experiment,
        "data": os.path.join(args.data_root, args.train_template.format(name=material.name)),
        "test_data": os.path.join(args.data_root, args.test_template.format(name=material.name)),
        "prefix": material.name,
        "xstart": crop.xstart,
        "ystart": crop.ystart,
        "xrange": crop.width,
        "yrange": crop.height,
        "offset_mode": args.offset_mode,
        "savedir": args.savedir or datetime.now().strftime("%Y%m%d"),
        "ending": ending,
    }

    argv: list[str] = []
    for key, value in settings.items():
        argv += [f"--{key}", str(value)]
    argv += ["--gt", "--plot"]
    # Anything unrecognised here is passed straight to train.py, and wins because
    # argparse keeps the last occurrence of an option.
    return argv + list(extra or [])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_idx", type=int, default=0, help=f"material, 0..{len(MATERIALS) - 1}")
    p.add_argument("--exp_idx", type=int, default=0, help=f"experiment, 0..{len(EXPERIMENTS) - 1}")
    p.add_argument("--data_root", default="data", help="directory holding the HDF5 captures")
    p.add_argument("--train_template", default=TRAIN_TEMPLATE)
    p.add_argument("--test_template", default=TEST_TEMPLATE)
    p.add_argument("--offset_mode", default="2d", choices=["2d", "1d"])
    p.add_argument("--preset", default="512", choices=PRESETS,
                   help="512: one tile, pyramid 128-512, 20k steps. "
                        "2k: the whole capture, pyramid scaled to it, 150k steps")
    p.add_argument("--capture_size", type=int, default=0,
                   help="capture width for --preset 2k (default: read it from the file)")
    p.add_argument("--crop_size", type=int, default=0, help="override the material's crop size")
    p.add_argument("--crop_index", type=int, default=None, help="override the material's tile")
    p.add_argument("--savedir", default="", help="output root (default: today's date)")
    p.add_argument("--list", action="store_true", help="print the material/experiment tables")
    p.add_argument("--dry_run", action="store_true", help="print the arguments and exit")
    args, extra = p.parse_known_args()  # unknown options are forwarded to train.py

    if args.list:
        print("materials:")
        for i, m in enumerate(MATERIALS):
            print(f"  {i:2d}  {m.name:<34} crop {m.crop_size}px, tile {m.crop_index}")
        print("experiments:")
        for i, e in enumerate(EXPERIMENTS):
            print(f"  {i:2d}  {e}")
        print("presets:")
        print(f"  512  one {COMMON['texture_res']}px tile, pyramid "
              f"{COMMON['min_res']}-{COMMON['texture_res']}, batch {COMMON['batch_size']}, "
              f"{COMMON['n_steps']} steps")
        print(f"   2k  the whole capture, pyramid size/4-size, offset texture size/16, "
              f"batch {PRESET_2K['batch_size']}, {PRESET_2K['n_steps']} steps")
        return

    if not 0 <= args.data_idx < len(MATERIALS):
        p.error(f"--data_idx must be in [0, {len(MATERIALS) - 1}]")
    if not 0 <= args.exp_idx < len(EXPERIMENTS):
        p.error(f"--exp_idx must be in [0, {len(EXPERIMENTS) - 1}]")

    argv = build_argv(args, extra)
    print("train.py " + " ".join(argv))
    if not args.dry_run:
        train_main(argv)


if __name__ == "__main__":
    main()
