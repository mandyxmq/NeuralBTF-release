#!/usr/bin/env python3
"""Score every trained material and collect the results into one table.

Point ``--savedir`` at the directory ``training/train_all.py`` wrote to.  For each
material the newest checkpoint below ``<savedir>/<material>/`` is evaluated on the
same crop it was trained on, and the per-material metrics are written next to that
checkpoint plus a summary table in ``--outdir``.

    python training/evaluate_all.py --savedir results
    python training/evaluate_all.py --savedir results --data_idx 0 --save_png /tmp/vis
    python training/evaluate_all.py --savedir results --splits val test --lpips
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), HERE]  # repo root (for neuralbtf) and this folder

from evaluate import SPLITS, build_parser, evaluate  # noqa: E402
from neuralbtf.data import Crop, capture_shape  # noqa: E402
from neuralbtf.materials import MATERIALS, TEST_TEMPLATE, TRAIN_TEMPLATE, crop_for  # noqa: E402
from neuralbtf.metrics import COLUMNS  # noqa: E402

ITERATION = re.compile(r"_iter_(\d+)\.pth$")


def find_checkpoint(savedir: str, name: str, pattern: str) -> str | None:
    """The highest-iteration checkpoint of ``name`` below ``savedir``."""
    matches = glob.glob(os.path.join(savedir, pattern.format(name=name)), recursive=True)
    if not matches:
        return None
    return max(matches, key=lambda p: int(ITERATION.search(p).group(1)) if ITERATION.search(p) else -1)


def write_table(rows: list[dict], columns: list[str], splits: list[str], outdir: str) -> None:
    """Write the summary as CSV and Markdown, with a mean over materials at the end."""
    os.makedirs(outdir, exist_ok=True)
    fields = ["material"] + [f"{s}_{c}" for s in splits for c in columns]

    mean_row = {"material": "mean"}
    for f in fields[1:]:
        values = [r[f] for r in rows if r.get(f) is not None]
        mean_row[f] = sum(values) / len(values) if values else None
    table = rows + [mean_row]

    csv_path = os.path.join(outdir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    def cell(value) -> str:
        return "-" if value is None else f"{value:.4f}"

    md_path = os.path.join(outdir, "metrics_summary.md")
    with open(md_path, "w") as fh:
        fh.write("| " + " | ".join(fields) + " |\n")
        fh.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for row in table:
            fh.write("| " + " | ".join([row["material"]]
                                       + [cell(row.get(f)) for f in fields[1:]]) + " |\n")

    print(f"[table] {md_path}")
    print(f"[table] {csv_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--savedir", required=True, help="output root used by train_all.py")
    p.add_argument("--data_idx", type=int, default=None,
                   help=f"evaluate one material, 0..{len(MATERIALS) - 1} (default: all)")
    p.add_argument("--data_root", default="data", help="directory holding the HDF5 captures")
    p.add_argument("--train_template", default=TRAIN_TEMPLATE)
    p.add_argument("--test_template", default=TEST_TEMPLATE)
    p.add_argument("--checkpoint_pattern", default="{name}/**/result/{name}_iter_*.pth",
                   help="glob for a material's checkpoint, relative to --savedir")
    p.add_argument("--preset", default="512", choices=["512", "2k"],
                   help="which run to score: 512 = the material's tile, "
                        "2k = the whole capture (matches train_all.py --preset)")
    p.add_argument("--crop_size", type=int, default=0, help="override the material's crop size")
    p.add_argument("--crop_index", type=int, default=None, help="override the material's tile")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--outdir", default="", help="where to write the summary table "
                                                "(default: --savedir)")
    p.add_argument("--save_png", default="", help="root directory for qualitative PNGs; "
                                                  "each material gets a subdirectory")
    p.add_argument("--lpips", action="store_true", help="also report LPIPS")
    args, extra = p.parse_known_args()  # unknown options are forwarded to evaluate.py

    materials = MATERIALS if args.data_idx is None else [MATERIALS[args.data_idx]]
    columns = list(COLUMNS) if args.lpips else list(COLUMNS[:-1])
    rows: list[dict] = []

    for material in materials:
        checkpoint = find_checkpoint(args.savedir, material.name, args.checkpoint_pattern)
        if checkpoint is None:
            print(f"[skip] {material.name}: no checkpoint under "
                  f"{os.path.join(args.savedir, material.name)}")
            continue

        train_path = os.path.join(args.data_root, args.train_template.format(name=material.name))
        if args.preset == "2k":  # the model was fitted to the whole capture
            _, height, width = capture_shape(train_path)
            crop = Crop(0, 0, width, height)
        else:
            crop = crop_for(material.name, args.crop_size, args.crop_index)

        argv = [
            "--checkpoint", checkpoint,
            "--data", train_path,
            "--test_data", os.path.join(args.data_root,
                                        args.test_template.format(name=material.name)),
            "--prefix", material.name,
            "--xstart", str(crop.xstart), "--ystart", str(crop.ystart),
            "--xrange", str(crop.width), "--yrange", str(crop.height),
            "--splits", *args.splits,
        ]
        if args.lpips:
            argv.append("--lpips")
        if args.save_png:
            argv += ["--save_png", os.path.join(args.save_png, material.name)]

        print(f"\n=== {material.name} ===")
        report = evaluate(build_parser().parse_args(argv + extra))

        row = {"material": material.name}
        for split in args.splits:
            means = report["splits"].get(split, {}).get("mean", {})
            row.update({f"{split}_{c}": means.get(c) for c in columns})
        rows.append(row)

    if rows:
        write_table(rows, columns, args.splits, args.outdir or args.savedir)
    else:
        print("[warn] nothing evaluated")


if __name__ == "__main__":
    main()
