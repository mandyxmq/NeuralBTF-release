# Latent-space synthesis

This pipeline extends a trained neural BTF beyond its captured spatial domain.
It is organized as five stages:

1. extract the latent stack from a trained BTF;
2. train a per-material latent diffusion model;
3. construct a 16-tile Wang set in latent space;
4. inpaint graph-cut seams with RePaint;
5. assemble and render a Wang tiling.

Each command is in `synthesis/`. Main components and data loaders are under `neuralbtf.synthesis`. Every stage outputs data and optional visualizations, so it can be inspected and rerun independently.

## Installation

Set up the base repository environment as described in the main README, then install the synthesis dependencies:

```bash
python -m pip install -e ".[synthesis]"
```

Synthesis starts from a trained `.pth` BTF checkpoint. The captured data is not required after BTF training.

## Example: pink hand towel

After pink hand towel BTF training, run all commands from the repository root:

```bash
EXPERIMENT=results/published_2k/pink_hand_towel_02/multitextureoffsetsph_1800_450_8_lr_1e-3_512_3_sph_112_8_1_512
CHECKPOINT="$EXPERIMENT/result/pink_hand_towel_02_iter_128640.pth"
PROJECT="$EXPERIMENT/synthesis/project.json"

python synthesis/extract_latents.py \
  --checkpoint "$CHECKPOINT" \
  --config synthesis/configs/default.json

python synthesis/train_diffusion.py --config "$PROJECT"
python synthesis/build_wang_tiles.py --config "$PROJECT"
python synthesis/repaint_wang_tiles.py --config "$PROJECT"
python synthesis/render_tiles.py --config "$PROJECT"
```

Extraction infers `<experiment>/synthesis` from the checkpoint path and creates `project.json`. 
All later stages use that generated project file. Use `--overwrite` only when intentionally replacing an existing stage.

## Configuration

`synthesis/configs/default.json` is the bootstrap template used during extraction. The generated `<experiment>/synthesis/project.json` is the per-material configuration to edit for all later stages. 
Stage `config.resolved.json` files record completed runs and are not inputs to later commands.

| Stage | JSON section | Typical adjustments |
| --- | --- | --- |
| Extraction | `extract` | Latent resolution and checkpoint-comparison preview |
| Diffusion | `diffusion.data`, `diffusion.model`, `diffusion.train`, `diffusion.preview` | Crop sampling, zoom range, model capacity, training length, and previews |
| Wang tiles | `wang_tiles.periodicity`, `wang_tiles.tile`, `wang_tiles.search`, `wang_tiles.visualization` | Tile size selection, period detection, graph-cut search, and previews |
| RePaint | `repaint.mask`, `repaint.process`, `repaint.visualization` | Cross-mask shape, sampling schedule, precision, and stage previews |
| Final tiling | `tiling` | Source tile set, grid size, seed, periodic boundaries, and preview |

Each visualization section has an `enabled` field. Preview renders use the fixed light and view directions recorded during extraction, which keeps images comparable across stages.

The generated RePaint coarse and fine zooms initially follow the maximum and minimum diffusion training zooms. 
If `diffusion.data.zoom_range` is changed in `project.json`, update `repaint.process.coarse_zoom` and `repaint.process.fine_zoom` to the same endpoints. 
RePaint validates this relationship before processing tiles.

## Workflow

### 1. Extract latents

`extract_latents.py` loads the trained BTF, combines its spatial feature levels and neural-offset features into one latent array, and records the channel layout and checkpoint provenance. 
It also renders the extracted stack and the source checkpoint under the same fixed directions and records their difference (should be tiny). 

### 2. Train diffusion

`train_diffusion.py` trains one zoom-conditioned diffusion model directly on random, non-wrapped crops of the material latent. No VAE is used. 
Training prints loss, elapsed time, ETA, and CUDA memory, and can save fixed-direction previews and a loss plot after configured epochs.

Adjust training and preview values in the `diffusion` section of `project.json`. 
For a quick test, reduce `diffusion.data.samples_per_epoch` or `diffusion.train.epochs`, or set `diffusion.train.max_steps`. 
The stage keeps the final EMA model as `diffusion/model_final.pt` and records the run in `diffusion/metadata.json` and `diffusion/history.json`.

### 3. Build Wang tiles

`build_wang_tiles.py` estimates latent periodicity, selects a rectangular tile shape, searches for compatible source regions, and combines graph-cut corners into all 16 Wang edge combinations.

Use `wang_tiles.tile` to choose automatic or manual dimensions. Use `wang_tiles.search` to tune the appearance and offset weights or shorten a test tile search. 
Inspect the periodicity, search-history, seam, and tile-set images in `visualization/wang_tiles` before repainting.

### 4. Repaint seams

`repaint_wang_tiles.py` builds a tapered cross mask from each graph-cut seam. It first repairs coarse structure at the largest trained zoom, then restores full-resolution detail at the smallest trained zoom while preserving context and Wang edge labels.

Test one configured tile before a full run:

```bash
python synthesis/repaint_wang_tiles.py \
  --config "$PROJECT" \
  --first-tile-only
```

The stage preview names include the tile ID and the `raw`, `coarse`, or `fine` stage. 
After inspection, run the full command with `--overwrite`.
Adjust mask geometry, sampling, precision, and the preview tile in the `repaint` section of `project.json`.

### 5. Assemble a tiling

`render_tiles.py` tiles the latent sets stochastically and writes it as a larger latent, and optionally renders a bounded-resolution preview. 
Set `tiling.source` to `repaint` for the final result or `raw` for graph-cut tiles before inpainting.
Change the grid dimensions, seed, and periodic-boundary option in the `tiling` section.

## Outputs

All stages share `<experiment>/synthesis`:

| Path | Contents |
| --- | --- |
| `project.json` | Per-material settings for the complete pipeline |
| `latent/` | Extracted latent, layout, source provenance, and validation |
| `diffusion/` | Final diffusion model, normalization, history, and metadata |
| `wang_tiles/` | Raw tile latents, seams, period analysis, and tile manifest |
| `repaint/` | Repainted tile latents and tile manifest |
| `tilings/` | Assembled latent, Wang grid, and metadata |
| `visualization/<stage>/` | Optional RGB, offset, mask, seam, and progress images |

## Tests

The synthesis test suite runs on CPU:

```bash
python tests/test_synthesis.py
python tests/test_synthesis_diffusion.py
python tests/test_synthesis_wang.py
python tests/test_synthesis_repaint.py
python tests/test_neuralbtf.py
```