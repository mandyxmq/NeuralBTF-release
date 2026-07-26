# Neural BTF

Compresses a measured **BTF** (bidirectional texture function) into a
multi-resolution neural texture with a view-dependent neural offset and a small MLP
decoder.

Given a stack of images of a material under many light/view directions, training
fits a function

```
(u, v, light_x, light_y, view_x, view_y)  ->  (R, G, B)
```

which can then be evaluated at any uv and any direction pair.

## Layout

```
neuralbtf/     the model and everything shared: textures, SH encoding, data reading,
               losses, metrics, EXR/PNG writers
training/      fit a BTF to a capture and score it        -> training/README.md
synthesis/     extend a trained model to a larger region  -> synthesis/README.md
tests/         self-contained checks, no pytest needed
data/          the captures (see below)
```

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
pip install -e .        # optional: makes `import neuralbtf` work outside the checkout
```

Dependencies are **torch, numpy and h5py**. matplotlib is optional (`--plot`), as is
`lpips` (`evaluate.py --lpips`). No CUDA compiler and no renderer are needed: the
spherical-harmonics encoding and the EXR and PNG writers are implemented here, in
about a hundred lines each.

Verify the install (runs on CPU in a few seconds):

```bash
python tests/test_neuralbtf.py
```

Run the scripts from the repository root; they add it to `sys.path` themselves, so
`pip install -e .` is only needed by code living outside this checkout.

## Data

Download the captures from
**[Google Drive](https://drive.google.com/drive/folders/1Yi0-Xyf5Qif_pK6mtpDTj1H0PL2kzLt8?usp=sharing)**
and unpack them into
`data/` next to this README:

```
data/
├── real_red_leather_08_full_factor1.hdf5      # dense direction sweep, for fitting
├── real_red_leather_08_random_factor1.hdf5    # held-out directions, for validation
├── real_sari_05_full_factor1.hdf5
└── ...
```

Each file holds a stack of images of one material:

| dataset | shape | meaning |
| --- | --- | --- |
| `ground_light` | `(N, 2)` | light direction, xy of a unit vector; z is reconstructed as `sqrt(1 - x² - y²)` |
| `ground_camera_dir` | `(N, 2)` | view direction, same convention |
| `ground_color` | `(N, H, W, 3)` | measured radiance, linear float32 |

File names use the original capture names; the paper and the results website use
numbered display names.  [`MATERIALS.md`](MATERIALS.md) maps one onto the other.

A capture is split by *direction*, not by pixel:

| split | directions | used for |
| --- | --- | --- |
| train | all of `_full_` (1062, a regular grid of light/view angles) | fitting |
| val | first half of `_random_` (~46, randomised angles) | the loss curve during training |
| test | second half of `_random_` (~47) | scored once, after training |

`--val_split` sets the fraction of the held-out file used for validation (default
`0.5`, the `nval = numdir // 2` convention of the evaluation scripts). Pass
`--val_split 1.0` to validate on the whole held-out file and skip the test set.

Only the requested crop and the first `--train_len` directions are read from disk, so
a 53 GB capture costs ~3.3 GB of RAM (1062 directions of a 512² crop). The read is a
strided window, so it is bound by random-access throughput: ~3 min from an external
USB disk, seconds from local NVMe. It happens once, before training starts.

> **Near-field captures are a different format** and are not supported here: their
> `ground_camera_dir` is `(N, H, W, 2)`, one view direction per *pixel*. Passing one
> raises an explanatory error rather than silently misinterpreting the data.

## Train

The batch driver holds the per-material crops and the experiment grid, and reads from
`data/`:

```bash
python training/train_all.py --list                    # material / experiment tables
python training/train_all.py --data_idx 0 --dry_run    # show resolved arguments
python training/train_all.py --data_idx 0              # train one material (30-45 min)
for i in $(seq 0 16); do python training/train_all.py --data_idx $i; done   # all of them
```

Or drive `train.py` yourself:

```bash
python training/train.py \
  --data      data/real_red_leather_08_full_factor1.hdf5 \
  --test_data data/real_red_leather_08_random_factor1.hdf5 \
  --prefix red_leather_08 \
  --xstart 700 --ystart 350 --xrange 350 --yrange 350 \
  --n_steps 20000 --batch_size 10 --lr_str 1e-3 \
  --savedir results --ending run1 --gt --plot
```

Each step draws `--batch_size` captured images and one jittered sample per cell of
an `--xnum` × `--ynum` grid over the crop, so a step sees `batch_size · xnum · ynum`
radiance samples. `python training/train.py --help` lists every option.

The table divides each capture into a 4×4 grid of tiles of `crop_size` (a quarter of
the capture width) and trains on one tile per material. A tile that falls outside
the image is reported as an error rather than silently shrunk; override with
`--crop_size` / `--crop_index`, or pass the crop directly to `train.py`.

Any option the driver does not recognise is forwarded to `train.py`, so
`--data_idx 3 --n_steps 500 --train_len 64` overrides the table for a quick run.

### Fitting a whole capture: `--preset 2k`

```bash
python training/train_all.py --data_idx 1 --preset 2k
```

instead of one tile, this fits the **entire** capture, with the texture pyramid
scaled to the capture's own size — read from the file, so the materials whose
captures are smaller than 2048² are handled automatically:

| capture | pyramid | offset texture | materials |
| --- | --- | --- | --- |
| 2048² | 512 → 2048 | 128 | 11 |
| 2000² | 500 → 2000 | 125 | gold_flowers_on_stripes_01 |
| 1800² | 450 → 1800 | 112 | pink_hand_towel_02 |
| 1600² | 400 → 1600 | 100 | silky_smooth_green_01 |
| 1400² | 350 → 1400 | 87 | red_leather_08, light_grey_satin_02 |

The preset also drops to one image per step and runs 150 000 steps; the model,
the loss, the split and the 512×512 uv sampling are unchanged. It validates every
5000 steps rather than every 100: a validation pass renders every held-out
direction over the whole capture (~9 s for 46 directions of 1400², against 17 ms
per training step), so the tile cadence would put 85% of the run into monitoring.
The final val and test numbers are computed in full regardless.

Score it with the matching flag, which evaluates over the whole capture rather
than the tile:

```bash
python training/evaluate_all.py --savedir <savedir> --data_idx 1 --preset 2k
```

**This needs memory.** The whole capture is held in RAM: ~53 GB for 1062
directions of 2048², ~25 GB at 1400². Lower `--train_len` if that does not fit
(`--train_len 500` halves it). Expect roughly an hour of compute for the full
schedule on a recent GPU, plus the one-off read.

## Evaluate

`evaluate.py` scores a checkpoint on the splits above and writes the numbers next to
it:

```bash
python training/evaluate.py \
  --checkpoint results/red_leather_08/<run>/result/red_leather_08_iter_20000.pth \
  --data      data/real_red_leather_08_full_factor1.hdf5 \
  --test_data data/real_red_leather_08_random_factor1.hdf5 \
  --xstart 700 --ystart 350 --xrange 350 --yrange 350 \
  --splits val test --save_png /tmp/vis --image_gap 10
```

Every direction of a split is rendered at pixel centres over the crop and compared
with the measured image, giving **PSNR**, **PSNR in `log1p` space** and **relative
L2** — per direction in `<prefix>_metrics.npz`, averaged per split in
`<prefix>_metrics.json`.  `--lpips` adds the paper table's fourth column and is the
one thing that needs an extra package (`pip install lpips`).  `--save_png` writes
ground truth, prediction and a brightened `|difference|` as sRGB PNGs; `--save_exr`
writes the same images as linear HDR.

The model architecture comes from the `*_model_config.json` written during training,
or, if there is none, is inferred from the checkpoint's tensor shapes — so a bare
`state_dict` can be scored as well.

The batch driver evaluates every material against the crops in the table and collects
the results into one table:

```bash
python training/evaluate_all.py --savedir results                       # -> metrics_summary.md / .csv
python training/evaluate_all.py --savedir results --data_idx 0 --lpips  # one material
```

## Output

```
<savedir>/<prefix>/multitextureoffsetsph_<texture_res>_<min_res>_<channels>_lr_<lr>_<ending>/
├── result/
│   ├── <prefix>_iter_<n_steps>.pth      # final weights (plain state_dict)
│   ├── <prefix>_latest.pth              # rolling checkpoint, every --ckpt_gap steps
│   ├── <prefix>_model_config.json       # arguments needed to rebuild the model
│   ├── <prefix>_final_metrics.json      # val and test loss, and each split's size
│   ├── <prefix>_loss.npy                # training loss per step
│   ├── <prefix>_val_loss.npy            # validation loss per --val_gap steps
│   ├── <prefix>_smooth_epoch_loss.npy   # training loss per epoch
│   ├── <prefix>_metrics.json            # written by evaluate.py: per-split means
│   └── <prefix>_metrics.npz             # written by evaluate.py: per-direction rows
└── image/
    ├── color_<i>_pred.exr / color_<i>_gt.exr            # training directions
    ├── color_<i>_val_pred.exr / color_<i>_val_gt.exr    # validation directions
    └── color_<i>_test_pred.exr / ..._test_gt.exr        # test directions
```

`<i>` is the direction's index in its source file, so test images start at `nval`.

Reload a trained model:

```python
import json, torch
from neuralbtf import ModelConfig, MultiResNeuralTextureModel

cfg = ModelConfig(**json.load(open("result/red_leather_08_model_config.json")))
model = MultiResNeuralTextureModel(cfg).cuda().eval()
model.load_state_dict(torch.load("result/red_leather_08_iter_20000.pth"))

# uv in [0,1]^2, directions as xy of unit vectors
x = torch.tensor([[0.5, 0.5, 0.1, 0.2, -0.3, 0.4]], device="cuda")
rgb = model(x)
```

## Model

`neuralbtf/models.py`, ~150 lines:

1. **Neural offset** — a small `--off_texture_res` feature texture plus an MLP maps
   (surface features, view direction) to a uv displacement, gated by a learned
   scalar initialised to 0, so training starts from the un-warped model. This is
   what reproduces parallax and self-occlusion. `--offset_mode 1d` instead predicts
   a scalar displacement applied along the projected view direction.
2. **Texture pyramid** — levels from `--min_res` to `--texture_res` (doubling), each
   with `--texture_channels` features, sampled bilinearly at the warped uv and
   concatenated.
3. **Decoder** — the spatial features plus degree-3 spherical harmonics of both
   directions, through a `--n_neurons` × `--n_hidden_layers` ReLU MLP to RGB.

Textures train at `--texture_lr_scale` (default 10×) the decoder's learning rate.
Training uses mixed precision by default (`--no_amp` to disable).
