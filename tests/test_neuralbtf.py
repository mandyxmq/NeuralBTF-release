#!/usr/bin/env python3
"""Self-contained checks: ``python tests/test_neuralbtf.py`` (no pytest required)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "training")]

from neuralbtf import ModelConfig, MultiResNeuralTextureModel, read_exr, write_exr  # noqa: E402
from neuralbtf import materials as materials_table  # noqa: E402
from neuralbtf.config import load_model  # noqa: E402
from neuralbtf.data import (BTFSlice, make_model_input, pixel_center_uv,  # noqa: E402
                            split_held_out, stratified_uv)
from neuralbtf.encodings import SphericalHarmonicsEncoding, spherical_harmonics  # noqa: E402
from evaluate import default_prefix, evaluate_split  # noqa: E402
from neuralbtf.losses import btf_loss, sample_images  # noqa: E402
from neuralbtf.metrics import image_metrics, log_psnr, psnr, relative_l2, tonemap  # noqa: E402
from neuralbtf.png import to_display, write_png  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def sh_basis_is_orthonormal():
    """The SH polynomials must integrate to the identity Gram matrix over the sphere."""
    rng = np.random.default_rng(0)
    n = 400_000
    v = torch.tensor(rng.normal(size=(n, 3)), dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)

    y = spherical_harmonics(v, degree=4)
    gram = (y.T @ y) / n * (4 * np.pi)
    err = (gram - torch.eye(16, dtype=torch.float64)).abs().max().item()
    assert err < 0.02, f"SH basis is not orthonormal (max deviation {err:.4f})"


@check
def sh_encoding_shapes_and_rescale():
    enc = SphericalHarmonicsEncoding(n_dirs=2, degree=3)
    assert (enc.n_input_dims, enc.n_output_dims) == (6, 18)

    x = torch.rand(7, 6) * 2 - 1
    assert enc(x).shape == (7, 18)

    # rescale_input reproduces tiny-cuda-nn, which maps [0, 1] onto [-1, 1].
    plain = SphericalHarmonicsEncoding(n_dirs=2, degree=3, rescale_input=False)
    torch.testing.assert_close(enc(x), plain(x * 2 - 1))


@check
def uv_grids_are_row_major():
    uv = pixel_center_uv(width=4, height=3, device=torch.device("cpu"))
    assert uv.shape == (12, 2)
    # Row-major: the first 4 samples share a v and sweep u.
    torch.testing.assert_close(uv[:4, 1], torch.full((4,), 0.5 / 3))
    assert torch.all(uv[:4, 0].diff() > 0)

    jittered = stratified_uv(4, 3, torch.device("cpu"))
    assert jittered.shape == (12, 2)
    assert jittered.min() >= 0.0 and jittered.max() <= 1.0
    # One sample per cell: the i-th sample stays inside the i-th cell.
    assert torch.all((jittered - uv).abs() <= torch.tensor([0.5 / 4, 0.5 / 3]) + 1e-6)


@check
def model_input_pairs_uv_with_directions():
    uv = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    dirs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    x = make_model_input(uv, dirs)
    # Image-major: all uv samples of image 0, then all uv samples of image 1.
    assert x.shape == (4, 6)
    torch.testing.assert_close(x[0], torch.tensor([0.1, 0.2, 1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(x[1], torch.tensor([0.3, 0.4, 1.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(x[2], torch.tensor([0.1, 0.2, 5.0, 6.0, 7.0, 8.0]))


@check
def sample_images_returns_pixels_in_order():
    """GT lookups use align_corners=True: uv 0..1 spans the first..last pixel centre."""
    images = torch.arange(2 * 3 * 4 * 3, dtype=torch.float32).reshape(2, 3, 4, 3)
    x = torch.linspace(0, 1, 4)
    y = torch.linspace(0, 1, 3)
    uv = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1).reshape(-1, 2)
    got = sample_images(images, uv.repeat(2, 1))
    torch.testing.assert_close(got, images.reshape(-1, 3))


@check
def losses_respond_to_error():
    pred = torch.rand(64, 3) + 0.1
    for mode in (0, 1, 2, 3, 4):
        exact = btf_loss(pred, pred.clone(), mode).item()
        wrong = btf_loss(pred, pred * 2, mode).item()
        # Modes 2 and 3 use a Charbonnier term with a 1e-3 floor.
        assert exact <= 1.1e-3, f"loss {mode} = {exact} on an exact match"
        assert wrong > exact, f"loss {mode} does not grow with error"


@check
def model_forward_and_offset_gate():
    for mode in ("2d", "1d"):
        cfg = ModelConfig(texture_res=32, min_res=8, texture_channels=4, offset_mode=mode,
                          off_texture_res=8, off_neurons=16, n_neurons=16, n_hidden_layers=2)
        model = MultiResNeuralTextureModel(cfg)
        assert model.resolutions == [8, 16, 32]

        x = torch.rand(32, 6)
        x[:, 2:] = x[:, 2:] * 1.2 - 0.6  # directions in [-0.6, 0.6]
        out = model(x)
        assert out.shape == (32, 3) and torch.isfinite(out).all()

        out.sum().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert not missing, f"no gradient reached {missing}"

        # The gate starts closed, so the uv lookup is unwarped at step 0.
        assert model.offset_gate.item() == 0.0
        with torch.no_grad():
            model.textures[0].normal_()
            model.offset_texture.normal_()
            base = model(x)
            model.offset_gate.fill_(0.5)
            warped = model(x)
        assert not torch.allclose(base, warped), "offset gate has no effect"


@check
def model_state_dict_names_match_reference():
    cfg = ModelConfig(texture_res=16, min_res=8, texture_channels=2, off_texture_res=4,
                      off_neurons=8, n_neurons=8, n_hidden_layers=2)
    names = set(MultiResNeuralTextureModel(cfg).state_dict())
    expected = {
        "offset_texture", "offset_gate",
        "offset_mlp.0.weight", "offset_mlp.0.bias", "offset_mlp.2.weight", "offset_mlp.2.bias",
        "textures.0", "textures.1",
        "mlp.model.0.weight", "mlp.model.0.bias",
        "mlp.model.2.weight", "mlp.model.2.bias",
        "mlp.model.4.weight", "mlp.model.4.bias",
    }
    assert names == expected, f"unexpected parameter names: {names ^ expected}"


@check
def texture_parameters_get_their_own_learning_rate():
    cfg = ModelConfig(texture_res=16, min_res=8, texture_channels=2, off_texture_res=4,
                      off_neurons=8, n_neurons=8, n_hidden_layers=2)
    groups = MultiResNeuralTextureModel(cfg).parameter_groups(lr=1e-3, texture_lr_scale=10.0)
    assert groups[0]["lr"] == 1e-2 and groups[1]["lr"] == 1e-3
    assert len(groups[0]["params"]) == 3  # two pyramid levels + the offset texture


@check
def exr_roundtrip():
    rng = np.random.default_rng(0)
    img = (rng.random((5, 7, 3)) * 100).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.exr")
        write_exr(path, img)
        np.testing.assert_array_equal(read_exr(path), img)


@check
def held_out_file_splits_into_val_and_test():
    """Training watches the first half of the held-out file; the rest is the test set."""
    n = 93  # red_leather_08's held-out capture
    data = BTFSlice(light=np.zeros((n, 2), np.float32), view=np.zeros((n, 2), np.float32),
                    color=np.arange(n * 2 * 2 * 3, dtype=np.float32).reshape(n, 2, 2, 3))
    val, test = split_held_out(data, 0.5)
    # The reference evaluation scripts use nval = numdir // 2.
    assert (val.n_dirs, test.n_dirs) == (46, 47)
    np.testing.assert_array_equal(val.color[0], data.color[0])
    np.testing.assert_array_equal(test.color[0], data.color[46])

    all_val, empty = split_held_out(data, 1.0)
    assert (all_val.n_dirs, empty.n_dirs) == (93, 0)

    try:
        split_held_out(data, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("val_split 0 should be rejected")


@check
def metrics_match_the_reference_formulas():
    gt = torch.rand(8, 8, 3) + 0.05
    pred = gt + 0.1

    torch.testing.assert_close(psnr(pred, gt), torch.tensor(-10.0 * np.log10(0.01**1)).float(),
                               rtol=1e-4, atol=1e-4)  # constant 0.1 error -> MSE 0.01
    torch.testing.assert_close(log_psnr(pred, gt),
                               psnr(torch.log1p(pred), torch.log1p(gt)))
    torch.testing.assert_close(relative_l2(pred, gt),
                               ((pred - gt) ** 2 / (pred**2 + 0.01)).mean())
    assert torch.isinf(psnr(gt, gt))  # an exact match has zero error

    # Negative radiance is clamped away before scoring, as in the reference scripts.
    values = image_metrics(torch.full_like(gt, -1.0), torch.zeros_like(gt))
    assert torch.isinf(torch.tensor(values["psnr"]))
    assert set(values) == {"psnr", "log_psnr", "rel_l2"}

    display = tonemap(torch.tensor([0.0, 1.0, 1e6]))
    assert display[0] == 0.0 and 0.0 < display[1] < 1.0 and display[2] <= 1.0


@check
def png_is_srgb_encoded_and_readable():
    """The PNG writer produces a valid file whose pixels survive a round trip."""
    rng = np.random.default_rng(0)
    img = rng.random((6, 5, 3)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.png")
        write_png(path, img)
        blob = open(path, "rb").read()
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"

        # Our IDAT is unfiltered 8-bit RGB, so it decodes with zlib alone.
        import struct
        import zlib
        pos, chunks = 8, {}
        while pos < len(blob):
            size = struct.unpack_from(">I", blob, pos)[0]
            kind = blob[pos + 4:pos + 8]
            chunks[kind] = blob[pos + 8:pos + 8 + size]
            pos += size + 12
        width, height, depth, color_type = struct.unpack_from(">IIBB", chunks[b"IHDR"], 0)
        assert (width, height, depth, color_type) == (5, 6, 8, 2)

        raw = np.frombuffer(zlib.decompress(chunks[b"IDAT"]), np.uint8).reshape(height, -1)
        assert np.all(raw[:, 0] == 0), "scanlines must use filter type 0"
        np.testing.assert_array_equal(raw[:, 1:].reshape(height, width, 3), to_display(img))

    # sRGB transfer: black stays black, white saturates, mid-grey lifts.
    checks = to_display(np.array([[[0.0, 1.0, 0.5]]], dtype=np.float32))[0, 0]
    assert checks[0] == 0 and checks[1] == 255 and abs(int(checks[2]) - 188) <= 1


@check
def material_table_matches_the_published_crops():
    assert len(materials_table.MATERIALS) == 17
    assert len(set(materials_table.NAMES)) == 17

    # crop_index 6 is the third tile of the second row.
    crop = materials_table.crop_for("red_leather_08")
    assert (crop.xstart, crop.ystart, crop.width, crop.height) == (700, 350, 350, 350)
    # yellow_vase_pattern_01 uses tile 3: last tile of the first row.
    crop = materials_table.crop_for("yellow_vase_pattern_01")
    assert (crop.xstart, crop.ystart) == (1536, 0)
    # red_gold_cloudy_temple_01 uses tile 13: second tile of the fourth row.
    crop = materials_table.crop_for("red_gold_cloudy_temple_01")
    assert (crop.xstart, crop.ystart) == (512, 1536)

    assert materials_table.get(0).name == "red_leather_08"
    assert materials_table.crop_for("sari_05", crop_size=256, crop_index=0).width == 256


@check
def full_capture_preset_matches_the_reference_driver():
    """`--preset 2k` must reproduce real/trainall_2k.py: res 4x tile, min_res tile, off tile//4."""
    for m in materials_table.MATERIALS:
        capture = m.crop_size * 4  # every tile is a quarter of its capture
        settings = materials_table.full_capture_settings(capture)
        crop = settings["crop"]
        assert (crop.xstart, crop.ystart, crop.width) == (0, 0, capture), m.name
        assert settings["texture_res"] == capture, m.name
        assert settings["min_res"] == m.crop_size, m.name
        assert settings["off_texture_res"] == m.crop_size // 4, m.name

    # red_leather_08's published 2k run: multitextureoffsetsph_1400_350_8_..._sph_87_8_1_512
    settings = materials_table.full_capture_settings(1400)
    assert (settings["texture_res"], settings["min_res"], settings["off_texture_res"]) \
        == (1400, 350, 87)

    # Non-square captures use the longer side; sizes that cannot tile are rejected.
    assert materials_table.full_capture_settings(2048, 1024)["texture_res"] == 2048
    try:
        materials_table.full_capture_settings(1023)
    except ValueError:
        pass
    else:
        raise AssertionError("a capture that does not divide by 4 should be rejected")


@check
def model_config_is_recoverable_from_a_checkpoint():
    """Checkpoints without a config JSON still load: the shapes describe the model."""
    for mode in ("2d", "1d"):
        cfg = ModelConfig(texture_res=32, min_res=8, texture_channels=4, offset_mode=mode,
                          off_texture_res=8, off_texture_channels=6, off_neurons=16,
                          off_hidden_layers=2, n_neurons=16, n_hidden_layers=2, sh_degree=3)
        state = MultiResNeuralTextureModel(cfg).state_dict()
        assert ModelConfig.from_state_dict(state) == cfg

    plain = ModelConfig(texture_res=16, min_res=16, texture_channels=2, use_neural_offset=False,
                        n_neurons=8, n_hidden_layers=1, sh_degree=2)
    assert ModelConfig.from_state_dict(MultiResNeuralTextureModel(plain).state_dict()) == plain


@check
def tiny_cuda_nn_checkpoints_load():
    """The original checkpoints carry an empty ``dir_encoder.params``; it is ignored."""
    cfg = ModelConfig(texture_res=16, min_res=8, texture_channels=2, off_texture_res=4,
                      off_neurons=8, n_neurons=8, n_hidden_layers=2)
    state = MultiResNeuralTextureModel(cfg).state_dict()
    state["dir_encoder.params"] = torch.empty(0)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "material_iter_20000.pth")
        torch.save(state, path)
        model, recovered = load_model(path, device="cpu")

    assert recovered == cfg
    assert default_prefix(path) == "material"
    assert not model.training


@check
def evaluate_split_scores_every_direction():
    cfg = ModelConfig(texture_res=16, min_res=8, texture_channels=2, off_texture_res=4,
                      off_neurons=8, n_neurons=8, n_hidden_layers=2)
    model = MultiResNeuralTextureModel(cfg).eval()

    rng = np.random.default_rng(0)
    n, h, w = 3, 5, 4
    data = BTFSlice(light=rng.normal(scale=0.3, size=(n, 2)).astype(np.float32),
                    view=rng.normal(scale=0.3, size=(n, 2)).astype(np.float32),
                    color=rng.random((n, h, w, 3)).astype(np.float32))
    uv = pixel_center_uv(w, h, torch.device("cpu"))

    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(eval_chunk=7, image_gap=2, save_png=tmp, save_exr="",
                                  exposure=0.0, diff_scale=5.0)
        rows = evaluate_split(model, data, uv, args, "test", index_offset=10)

        assert rows.shape == (n, 3) and np.isfinite(rows).all()
        # Directions 0 and 2 are saved, renamed by index_offset; each writes 3 images.
        written = sorted(os.listdir(tmp))
        assert written == [f"test_{i:04d}_{kind}.png"
                           for i in (10, 12) for kind in ("diff", "gt", "pred")], written


def run() -> int:
    failures = 0
    for fn in CHECKS:
        try:
            fn()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run())
