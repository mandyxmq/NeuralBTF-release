# Synthesis

Extends a trained neural BTF to a larger region than the crop it was fitted to.

*(Placeholder — the code lands here.)*

## What to build against

The model and everything around it is in [`neuralbtf/`](../neuralbtf), which is a
plain importable package. Run `pip install -e .` from the repository root, or start
your script with

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

as the scripts in [`training/`](../training) do.

Loading a trained material:

```python
from neuralbtf import load_model, crop_for, write_exr, write_png

model, cfg = load_model("…/red_leather_08_iter_20000.pth", device="cuda")
model.textures        # the pyramid: one (1, C, R, R) tensor per level, min_res..texture_res
model.offset_texture  # the neural-offset feature texture, (1, C, R, R)
model.offset_gate     # scalar gate on the offset
cfg.texture_res, cfg.min_res, cfg.texture_channels, cfg.offset_mode
```

Evaluating it, `(N, 6)` in -> `(N, 3)` out, uv in `[0, 1]²`, directions as the xy of
unit vectors:

```python
x = torch.tensor([[u, v, light_x, light_y, view_x, view_y]], device="cuda")
rgb = model(x)
```

`neuralbtf.data.pixel_center_uv`, `make_model_input`, `read_btf` and `Crop` cover the
usual "render this uv grid for this direction pair" plumbing;
`training/evaluate.py:render` is a worked example.

## Please keep

- **Parameter names.** Checkpoints load with `strict=True`; renaming a module or a
  parameter breaks every trained material, including the published ones.
- **The load path.** Use `load_model`, so a checkpoint keeps working whether or not it
  has a `*_model_config.json` beside it.
- **Shared code in `neuralbtf/`.** Anything training and synthesis both need belongs
  in the package, not duplicated in the two folders.
- **Tests in `tests/`.** `python tests/test_neuralbtf.py` runs on CPU in a few seconds
  and takes no pytest; adding `tests/test_synthesis.py` beside it keeps that true.
