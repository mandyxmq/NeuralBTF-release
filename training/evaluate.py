#!/usr/bin/env python3
"""Score a trained neural BTF against the capture it was fitted to.

Every direction of a split is rendered at pixel centres over the training crop and
compared with the measured image: PSNR, PSNR in ``log1p`` space, relative L2, and
optionally LPIPS.  The splits are the ones training used -- all of the ``_full_``
file, and the held-out ``_random_`` file cut in two by ``--val_split``.

    python training/evaluate.py --checkpoint ... --test_data ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralbtf.config import load_model  # noqa: E402
from neuralbtf.data import (BTFSlice, Crop, make_model_input, pixel_center_uv,  # noqa: E402
                            read_btf, split_held_out)
from neuralbtf.exr import write_exr  # noqa: E402
from neuralbtf.metrics import COLUMNS, LPIPS, image_metrics  # noqa: E402
from neuralbtf.png import write_png  # noqa: E402

SPLITS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained neural BTF checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model = p.add_argument_group("model")
    model.add_argument("--checkpoint", required=True, help="trained *.pth (a plain state_dict)")
    model.add_argument("--model_config", default="",
                       help="architecture JSON; default: the *_model_config.json next to "
                            "the checkpoint, else inferred from the checkpoint itself")

    data = p.add_argument_group("data")
    data.add_argument("--data", default="", help="training HDF5 (only needed for --splits train)")
    data.add_argument("--test_data", required=True, help="held-out HDF5 (val + test directions)")
    data.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS,
                      help="which splits to score")
    data.add_argument("--val_split", type=float, default=0.5,
                      help="fraction of the held-out file that is validation; the rest is test")
    data.add_argument("--train_len", type=int, default=0, help="max training directions (0 = all)")
    data.add_argument("--test_len", type=int, default=0, help="max held-out directions (0 = all)")
    data.add_argument("--scale", type=float, default=1.0, help="multiplier applied to radiance")
    data.add_argument("--xstart", type=int, default=0, help="crop origin x")
    data.add_argument("--ystart", type=int, default=0, help="crop origin y")
    data.add_argument("--xrange", type=int, default=512, help="crop width")
    data.add_argument("--yrange", type=int, default=512, help="crop height")

    out = p.add_argument_group("output")
    out.add_argument("--prefix", default="", help="name for the output files "
                                                  "(default: from the checkpoint filename)")
    out.add_argument("--outdir", default="", help="where to write metrics "
                                                  "(default: next to the checkpoint)")
    out.add_argument("--save_png", default="", help="directory for GT/pred/diff PNGs")
    out.add_argument("--save_exr", default="", help="directory for GT/pred EXRs (linear HDR)")
    out.add_argument("--image_gap", type=int, default=100,
                     help="save images for every Nth direction of each split (0 = none)")
    out.add_argument("--exposure", type=float, default=0.0, help="exposure stops for PNGs")
    out.add_argument("--diff_scale", type=float, default=5.0,
                     help="brightening applied to the |pred - gt| PNGs")
    out.add_argument("--lpips", action="store_true",
                     help="also report LPIPS (needs the optional 'lpips' package)")
    out.add_argument("--eval_chunk", type=int, default=512 * 512,
                     help="pixels per forward pass")
    out.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p


def default_prefix(checkpoint: str) -> str:
    """``red_leather_08_iter_20000.pth`` -> ``red_leather_08``."""
    name = os.path.splitext(os.path.basename(checkpoint))[0]
    for marker in ("_iter_", "_latest", "_final"):
        if marker in name:
            return name.split(marker)[0]
    return name


@torch.no_grad()
def render(model, uv: torch.Tensor, direction: torch.Tensor, height: int, width: int,
           chunk: int) -> torch.Tensor:
    """Render one direction pair over the whole crop. Returns ``(H, W, 3)``."""
    out = torch.empty((uv.shape[0], 3), device=uv.device, dtype=torch.float32)
    for i in range(0, uv.shape[0], chunk):
        out[i:i + chunk] = model(make_model_input(uv[i:i + chunk], direction.view(1, 4))).float()
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.view(height, width, 3)


def _save_images(pred: torch.Tensor, gt: torch.Tensor, tag: str, index: int, args) -> None:
    if args.save_exr:
        os.makedirs(args.save_exr, exist_ok=True)
        write_exr(os.path.join(args.save_exr, f"{tag}_{index:04d}_pred.exr"),
                  pred.clamp(min=0.0).cpu().numpy())
        write_exr(os.path.join(args.save_exr, f"{tag}_{index:04d}_gt.exr"), gt.cpu().numpy())

    if args.save_png:
        os.makedirs(args.save_png, exist_ok=True)
        diff = (pred.clamp(min=0.0) - gt.clamp(min=0.0)).abs() * args.diff_scale
        for name, image in (("gt", gt), ("pred", pred), ("diff", diff)):
            write_png(os.path.join(args.save_png, f"{tag}_{index:04d}_{name}.png"),
                      image.clamp(min=0.0).cpu().numpy(), exposure_stops=args.exposure)


@torch.no_grad()
def evaluate_split(model, data: BTFSlice, uv: torch.Tensor, args, tag: str,
                   lpips: LPIPS | None = None, index_offset: int = 0) -> np.ndarray:
    """Metrics for every direction of ``data``. Returns ``(N, len(columns))``."""
    device = uv.device
    columns = COLUMNS if lpips is not None else COLUMNS[:-1]
    rows = np.empty((data.n_dirs, len(columns)), dtype=np.float32)
    dirs = torch.from_numpy(data.dirs).to(device)

    for k in range(data.n_dirs):
        pred = render(model, uv, dirs[k], data.height, data.width, args.eval_chunk)
        gt = torch.from_numpy(data.color[k]).to(device)

        values = image_metrics(pred, gt, lpips)
        rows[k] = [values[c] for c in columns]

        if args.image_gap > 0 and k % args.image_gap == 0:
            _save_images(pred, gt, tag, k + index_offset, args)

    return rows


def summarise(rows: np.ndarray, columns: tuple[str, ...]) -> dict:
    return {
        "n_dirs": int(rows.shape[0]),
        "mean": {c: float(rows[:, i].mean()) for i, c in enumerate(columns)},
        "std": {c: float(rows[:, i].std()) for i, c in enumerate(columns)},
    }


def evaluate(args) -> dict:
    device = torch.device(args.device)
    model, cfg = load_model(args.checkpoint, device, args.model_config or None)

    lpips = LPIPS(device=device) if args.lpips else None
    columns = COLUMNS if lpips is not None else COLUMNS[:-1]
    crop = Crop(args.xstart, args.ystart, args.xrange, args.yrange)

    # Same split as training: all of --data trains, --test_data is cut into a
    # validation half (watched during training) and a test half (never seen).
    splits: dict[str, tuple[BTFSlice, int]] = {}
    if "train" in args.splits:
        if not args.data:
            raise ValueError("--splits train needs --data, the capture that was trained on")
        splits["train"] = (read_btf(args.data, crop, args.train_len or None, args.scale), 0)
    if {"val", "test"} & set(args.splits):
        held_out = read_btf(args.test_data, crop, args.test_len or None, args.scale)
        val_data, test_data = split_held_out(held_out, args.val_split)
        if "val" in args.splits:
            splits["val"] = (val_data, 0)
        if "test" in args.splits:
            splits["test"] = (test_data, val_data.n_dirs)

    prefix = args.prefix or default_prefix(args.checkpoint)
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(outdir, exist_ok=True)

    report = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data) if args.data else "",
        "test_data": os.path.abspath(args.test_data),
        "crop": {"xstart": crop.xstart, "ystart": crop.ystart,
                 "width": crop.width, "height": crop.height},
        "val_split": args.val_split,
        "model": cfg.as_dict(),
        "columns": list(columns),
        "splits": {},
    }
    per_direction = {}

    for tag in SPLITS:  # keep a stable order regardless of how --splits was given
        if tag not in splits:
            continue
        data, offset = splits[tag]
        if data.n_dirs == 0:  # e.g. --val_split 1.0 leaves no test directions
            print(f"[{tag:>5}] empty, skipped")
            continue
        uv = pixel_center_uv(data.width, data.height, device)
        rows = evaluate_split(model, data, uv, args, tag, lpips, offset)
        per_direction[tag] = rows
        report["splits"][tag] = summarise(rows, columns)
        means = report["splits"][tag]["mean"]
        print(f"[{tag:>5}] {data.n_dirs:4d} dirs  " +
              "  ".join(f"{c} {means[c]:.4f}" for c in columns))

    with open(os.path.join(outdir, f"{prefix}_metrics.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    np.savez(os.path.join(outdir, f"{prefix}_metrics.npz"),
             columns=np.array(columns), **per_direction)
    print(f"[done] {os.path.join(outdir, prefix)}_metrics.json / .npz")
    return report


def main(argv: list[str] | None = None) -> None:
    evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
