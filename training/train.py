#!/usr/bin/env python3
"""Train a neural BTF on a far-field capture.

    python training/train.py --data ... --test_data ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralbtf.config import add_model_arguments, model_config_from_args  # noqa: E402
from neuralbtf.data import (BTFImageDataset, BTFSlice, Crop, make_model_input,  # noqa: E402
                            pixel_center_uv, read_btf, split_held_out, stratified_uv)
from neuralbtf.exr import write_exr  # noqa: E402
from neuralbtf.losses import LOSS_MODES, btf_loss, training_loss  # noqa: E402
from neuralbtf.models import MultiResNeuralTextureModel  # noqa: E402

MODEL_TYPE_NAMES = {"2d": "multitextureoffsetsph", "1d": "multitextureoffsetsph_1d"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a multi-resolution neural texture BTF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("--data", required=True, help="training HDF5 (far-field BTF)")
    data.add_argument("--test_data", required=True, help="validation HDF5 (held-out directions)")
    data.add_argument("--prefix", default="btf", help="material name; used for output paths")
    data.add_argument("--train_len", type=int, default=2000, help="max training directions")
    data.add_argument("--test_len", type=int, default=0, help="max held-out directions (0 = all)")
    data.add_argument("--val_split", type=float, default=0.5,
                      help="fraction of the held-out file used for validation during "
                           "training; the remainder is the test set, which is only "
                           "scored once at the end (1.0 = validate on everything)")
    data.add_argument("--scale", type=float, default=1.0, help="multiplier applied to radiance")
    data.add_argument("--xstart", type=int, default=0, help="crop origin x")
    data.add_argument("--ystart", type=int, default=0, help="crop origin y")
    data.add_argument("--xrange", type=int, default=512, help="crop width")
    data.add_argument("--yrange", type=int, default=512, help="crop height")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--n_steps", type=int, default=20000, help="training steps")
    opt.add_argument("--batch_size", type=int, default=10, help="images per step")
    opt.add_argument("--xnum", type=int, default=512, help="uv samples per row, per image")
    opt.add_argument("--ynum", type=int, default=512, help="uv samples per column, per image")
    opt.add_argument("--lr_str", default="1e-3", help="base learning rate")
    opt.add_argument("--texture_lr_scale", type=float, default=10.0,
                     help="learning-rate multiplier for texture parameters")
    opt.add_argument("--loss", type=int, default=0, choices=sorted(LOSS_MODES),
                     help="; ".join(f"{k}: {v}" for k, v in LOSS_MODES.items()))
    opt.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    opt.add_argument("--num_workers", type=int, default=0, help="dataloader workers")
    opt.add_argument("--seed", type=int, default=1025)
    opt.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    add_model_arguments(p)

    out = p.add_argument_group("output")
    out.add_argument("--savedir", default="./", help="root output directory")
    out.add_argument("--ending", default="", help="suffix identifying this run")
    out.add_argument("--val_gap", type=int, default=100, help="steps between validations")
    out.add_argument("--log_gap", type=int, default=100, help="steps between log lines")
    out.add_argument("--ckpt_gap", type=int, default=1000, help="steps between checkpoint writes")
    out.add_argument("--gap", type=int, default=50, help="render every Nth training view (0 = none)")
    out.add_argument("--gap2", type=int, default=5, help="render every Nth validation view (0 = none)")
    out.add_argument("--gt", action="store_true", help="also write ground-truth EXRs")
    out.add_argument("--plot", action="store_true", help="write loss curves as PNG (needs matplotlib)")
    out.add_argument("--eval_chunk", type=int, default=512 * 512,
                     help="pixels per forward pass during validation and rendering")

    return p


@dataclass
class Paths:
    run_dir: str
    result_dir: str
    image_dir: str

    def __post_init__(self):
        for d in (self.result_dir, self.image_dir):
            os.makedirs(d, exist_ok=True)


def make_paths(args) -> Paths:
    """``savedir/prefix/<model>_<res>_<min_res>_<channels>_lr_<lr>_<ending>/``."""
    model_type = MODEL_TYPE_NAMES[args.offset_mode]
    configuration = f"{model_type}_{args.texture_res}_{args.min_res}_{args.texture_channels}"
    name = f"{configuration}_lr_{args.lr_str}"
    if args.ending:
        name += f"_{args.ending}"
    run_dir = os.path.join(args.savedir, args.prefix, name)
    return Paths(run_dir, os.path.join(run_dir, "result"), os.path.join(run_dir, "image"))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, uv: torch.Tensor, dirs: torch.Tensor, chunk: int) -> torch.Tensor:
    """Evaluate ``model`` for one direction pair over all of ``uv``. Returns ``(P, 3)``."""
    out = torch.empty((uv.shape[0], 3), device=uv.device, dtype=torch.float32)
    for i in range(0, uv.shape[0], chunk):
        x = make_model_input(uv[i:i + chunk], dirs.view(1, 4))
        out[i:i + chunk] = model(x).float()
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def validate(model, data: BTFSlice, uv: torch.Tensor, loss_mode: int, chunk: int,
             device: torch.device) -> float:
    """Mean loss over every pixel of every direction in ``data``."""
    if data.n_dirs == 0:
        return float("nan")

    was_training = model.training
    model.eval()

    dirs = torch.from_numpy(data.dirs).to(device)
    total, n = 0.0, 0
    for i in range(data.n_dirs):
        pred = predict(model, uv, dirs[i], chunk)
        target = torch.from_numpy(data.color[i]).to(device).reshape(-1, 3)
        total += btf_loss(pred, target, loss_mode).item() * pred.shape[0]
        n += pred.shape[0]

    model.train(was_training)
    return total / max(n, 1)


@torch.no_grad()
def render_views(model, data: BTFSlice, uv: torch.Tensor, gap: int, chunk: int,
                 device: torch.device, image_dir: str, tag: str, write_gt: bool,
                 index_offset: int = 0) -> int:
    """Render every ``gap``-th direction of ``data`` to ``image_dir``.

    Filenames carry the direction's index in the source file, so ``index_offset``
    should be where ``data`` starts within it.
    """
    if gap <= 0 or data.n_dirs == 0:
        return 0

    was_training = model.training
    model.eval()

    dirs = torch.from_numpy(data.dirs).to(device)
    shape = (data.height, data.width, 3)
    written = 0
    for i in range(0, data.n_dirs, gap):
        name = f"color_{i + index_offset}{tag}"
        pred = predict(model, uv, dirs[i], chunk).clamp(min=0.0).reshape(shape)
        write_exr(os.path.join(image_dir, f"{name}_pred.exr"), pred.cpu().numpy())
        if write_gt:
            write_exr(os.path.join(image_dir, f"{name}_gt.exr"), data.color[i])
        written += 1

    model.train(was_training)
    return written


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _save_curve(path_prefix: str, name: str, values: list[float], plot: bool,
                title: str, xlabel: str) -> None:
    curve = np.asarray(values, dtype=np.float64)
    np.save(f"{path_prefix}_{name}.npy", curve)
    if not plot or curve.size == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure()
    plt.plot(curve)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(f"{path_prefix}_{name}.png")
    plt.close()


def train(args) -> str:
    start_time = time.time()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    paths = make_paths(args)
    out_prefix = os.path.join(paths.result_dir, args.prefix)
    print(f"[run] {paths.run_dir}")

    # --- Data -------------------------------------------------------------
    crop = Crop(args.xstart, args.ystart, args.xrange, args.yrange)
    train_data = read_btf(args.data, crop, args.train_len, args.scale)

    # The held-out file is split by direction: the first `val_split` fraction is the
    # validation set watched during training, the rest is the test set, scored once at
    # the end. Matches the `nval = numdir // 2` convention of the evaluation scripts.
    held_out = read_btf(args.test_data, crop, args.test_len or None, args.scale)
    val_data, test_data = split_held_out(held_out, args.val_split)
    n_val = val_data.n_dirs
    print(f"[data] held-out directions: {val_data.n_dirs} validation, "
          f"{test_data.n_dirs} test (never seen during training)")

    xnum = min(args.xnum, train_data.width)
    ynum = min(args.ynum, train_data.height)
    print(f"[data] crop {train_data.width}x{train_data.height}, "
          f"{xnum}x{ynum} uv samples per image per step")

    loader = DataLoader(BTFImageDataset(train_data), batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    if len(loader) == 0:
        raise ValueError(f"batch_size ({args.batch_size}) exceeds the number of training "
                         f"directions ({train_data.n_dirs})")

    # --- Model ------------------------------------------------------------
    cfg = model_config_from_args(args)
    model = MultiResNeuralTextureModel(cfg).to(device)
    print(model)
    print(f"[model] texture pyramid {model.resolutions}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    with open(f"{out_prefix}_model_config.json", "w") as fh:
        json.dump(cfg.as_dict(), fh, indent=2)

    lr = float(args.lr_str)
    optimizer = torch.optim.Adam(model.parameter_groups(lr, args.texture_lr_scale))
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"[optim] lr {lr} (textures x{args.texture_lr_scale}), amp={use_amp}, "
          f"loss={args.loss} ({LOSS_MODES[args.loss]})")

    # Pixel-centre uv grid, shared by validation and rendering.
    eval_uv = pixel_center_uv(train_data.width, train_data.height, device)

    # --- Loop ---------------------------------------------------------------
    steps_per_epoch = len(loader)
    print(f"[train] {args.n_steps} steps, {steps_per_epoch} steps/epoch, "
          f"~{args.n_steps / steps_per_epoch:.1f} epochs")

    step_losses: list[float] = []
    epoch_losses: list[float] = []
    val_losses: list[float] = []
    val_loss = float("nan")
    step = 0
    tick, val_time = time.perf_counter(), 0.0

    model.train()
    while step < args.n_steps:
        epoch_loss, n_batches = 0.0, 0

        for dirs, images, _ in loader:
            dirs = dirs.to(device, non_blocking=True)
            images = images.to(device, non_blocking=True)

            # Fresh jittered uv samples every step, shared across the batch.
            uv = stratified_uv(xnum, ynum, device)
            x = make_model_input(uv, dirs)
            uv_tiled = x[:, :2]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                pred = model(x)
                loss = training_loss(pred, images, uv_tiled, args.loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step_losses.append(loss.item())
            epoch_loss += step_losses[-1]
            n_batches += 1
            step += 1

            if args.val_gap > 0 and step % args.val_gap == 0:
                val_start = time.perf_counter()
                val_loss = validate(model, val_data, eval_uv, args.loss, args.eval_chunk, device)
                val_losses.append(val_loss)
                _save_curve(out_prefix, "val_loss", val_losses, args.plot,
                            "Validation loss", "Validation steps")
                val_time += time.perf_counter() - val_start

            if args.log_gap > 0 and step % args.log_gap == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick - val_time
                window = step_losses[-args.log_gap:]
                print(f"[train] step {step}/{args.n_steps}  train {np.mean(window):.6f}  "
                      f"val {val_loss:.6f}  {elapsed * 1e3 / args.log_gap:.1f} ms/step")
                tick, val_time = time.perf_counter(), 0.0

            if args.ckpt_gap > 0 and step % args.ckpt_gap == 0:
                torch.save(model.state_dict(), f"{out_prefix}_latest.pth")

            if step >= args.n_steps:
                break

        if n_batches:
            epoch_losses.append(epoch_loss / n_batches)

    _save_curve(out_prefix, "loss", step_losses, args.plot, "Training loss", "Training steps")
    _save_curve(out_prefix, "smooth_epoch_loss", epoch_losses, args.plot,
                "Training loss (epoch mean)", "Epochs")

    checkpoint = f"{out_prefix}_iter_{step}.pth"
    torch.save(model.state_dict(), checkpoint)
    print(f"[train] saved {checkpoint}")

    # --- Final renders --------------------------------------------------------
    rendered = {
        "train": render_views(model, train_data, eval_uv, args.gap, args.eval_chunk, device,
                              paths.image_dir, "", args.gt),
        "val": render_views(model, val_data, eval_uv, args.gap2, args.eval_chunk, device,
                            paths.image_dir, "_val", args.gt),
        "test": render_views(model, test_data, eval_uv, args.gap2, args.eval_chunk, device,
                             paths.image_dir, "_test", args.gt, index_offset=n_val),
    }
    print(f"[render] {rendered} -> {paths.image_dir}")

    # --- Final metrics ----------------------------------------------------------
    metrics = {
        "loss_mode": args.loss,
        "steps": step,
        "train_loss": float(np.mean(step_losses[-steps_per_epoch:])),
        "val_loss": validate(model, val_data, eval_uv, args.loss, args.eval_chunk, device),
        "test_loss": validate(model, test_data, eval_uv, args.loss, args.eval_chunk, device),
        "n_dirs": {"train": train_data.n_dirs, "val": val_data.n_dirs, "test": test_data.n_dirs},
    }
    with open(f"{out_prefix}_final_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[done] val {metrics['val_loss']:.6f} ({val_data.n_dirs} dirs), "
          f"test {metrics['test_loss']:.6f} ({test_data.n_dirs} dirs), "
          f"total {time.time() - start_time:.1f}s")
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
