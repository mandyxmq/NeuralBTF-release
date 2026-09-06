# Material names

The 17 captured materials were renamed for the paper and the results website.  The
capture files, the checkpoints and the code all still use the **original** capture
names, so this table is the bridge between the two.

`#` is the material's number in the paper (`01. Light Grey Cloth`, …).  `--data_idx`
is what `training/train_all.py` and `training/evaluate_all.py` take, i.e. the position
in the `MATERIALS` table of `neuralbtf/materials.py`.

| # | New name (paper / website) | Old name (data files, code) | `--data_idx` |
| --- | --- | --- | --- |
| 1 | Light Grey Cloth | dark_blue_cloth_01 | 5 |
| 2 | Gold Floral Blue Cloth | goldflower_blue_cloth_01 | 6 |
| 3 | Olive Velvet | olive_green_velvet_01 | 2 |
| 4 | Brown Tiled Velvet | brown_velvet_tiles_01 | 8 |
| 5 | Green Curtain | green_curtain_01 | 4 |
| 6 | Silky Green Fabric | silky_smooth_green_01 | 16 |
| 7 | Geometric Circles | circles_01 | 7 |
| 8 | Curvy Brown Pattern | curvy_browns_01 | 12 |
| 9 | Gold Floral Stripes | gold_flowers_on_stripes_01 | 14 |
| 10 | Trees on Yellow Pattern | trees_on_yellow_01 | 13 |
| 11 | Vanilla Floral Chevrons | vanilla_flowers_on_chevrons_01 | 15 |
| 12 | Yellow Vase Pattern | yellow_vase_pattern_01 | 10 |
| 13 | Pink Hand Towel | pink_hand_towel_02 | 9 |
| 14 | Handwoven Sari | sari_05 | 1 |
| 15 | Red–Gold Temple Pattern | red_gold_cloudy_temple_01 | 11 |
| 16 | Red Leather | red_leather_08 | 0 |
| 17 | Light Gray Car Paint | light_grey_satin_02 | 3 |

The old name is also the file name: a material's capture is
`real_<old name>_full_factor1.hdf5` (training directions) and
`real_<old name>_random_factor1.hdf5` (held-out directions).

Per-material crops live in `neuralbtf/materials.py`; `python training/train_all.py --list`
prints them.

### Where this mapping comes from

`real/update_html.py` in the research tree (`NeuralBTF-tiny-cuda-nn`) holds the three
lists this table is built from: `datalist` (old names, the order the code uses),
`display_names_raw` (new names, same order) and
`sample_index = [5, 6, 2, 8, 4, 16, 7, 12, 14, 13, 15, 10, 9, 1, 11, 0, 3]`, which is
the paper's ordering.  The same mapping appears in `real/update_metrics_and_images.py`,
`real/update_predictions_only.py` and in `NeuralBTF-Neumip/reconvert_and_upload.py`,
and the numbering matches the published results page
`Submission/html/allmaterials_2k/index.html`.
