# Training and evaluation

Fits a neural BTF to one crop of a capture, and scores the result. The model itself
lives in [`neuralbtf/`](../neuralbtf); everything here is a command-line front end.

| file | what it does |
| --- | --- |
| `train.py` | fit one crop of one capture |
| `evaluate.py` | score a checkpoint: PSNR, log-PSNR, relative L2, optional LPIPS |
| `train_all.py` | the material and experiment tables; drives `train.py` |
| `evaluate_all.py` | finds each material's checkpoint and collects one summary table |
| `configs/` | the tiny-cuda-nn JSON the published runs used (only `network` is read) |

```bash
python training/train_all.py --list                    # tables and presets
python training/train_all.py --data_idx 0              # train material 0 on its tile
python training/train_all.py --data_idx 0 --preset 2k  # ... on the whole capture
python training/evaluate_all.py --savedir <savedir>    # score whatever it produced
```

Two presets: `512` fits one tile of the capture with a 128→512 pyramid for 20 000
steps, `2k` fits the whole capture with the pyramid scaled to its size (size/4 →
size, offset texture size/16) for 150 000 steps at one image per step. Pass the same
`--preset` to `evaluate_all.py`, so it scores over the pixels the run was fitted to.

Run them from the repository root, with the captures in `data/`. See the main
[README](../README.md) for the arguments, the train/val/test split, and the output
layout.
