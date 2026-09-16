# Data contract and changing input formats

## Expected portable-folder layout

Put source data below `data/raw/` exactly as follows for the supplied SimJEB
configuration:

```text
data/raw/
├── all_bracket_metadata.csv              # must contain at least an `id` column
└── full_dataset/
    ├── surfmesh_obj/
    │   ├── 0.obj
    │   ├── 1.obj
    │   └── ...
    └── simresults_csv/
        ├── 0field.csv
        ├── 1field.csv
        └── ...
```

FEM decks, STEP, H3D, images, and volume meshes are not inputs to this
surface-only experiment. Do not place generated graphs or model outputs inside
`data/raw/`.

## OBJ requirements

Each OBJ must contain 3D `v` vertex records and triangular or polygonal `f`
face records. Polygonal faces are fan-triangulated. Vertices must be in exactly
the same order as the rows of the matching field CSV where `surf != 0`.

The default coordinate tolerance is `0.001` in the source coordinate unit.
The preparation step checks both the vertex count and the largest coordinate
distance. It stops on a mismatch rather than guessing a correspondence.

## CSV requirements

The CSV can contain extra columns. It must have one row per solver node and
these logical fields:

| Logical field | Meaning |
|---|---|
| `surf` | Nonzero indicates a surface node present in the OBJ. Values 1–3 are retained as surface classes. |
| `x`, `y`, `z` | Node coordinates in the same frame/unit as the OBJ. |
| `{ver,hor,dia,tor}_xdisp` | Displacement X component for each load. |
| `{ver,hor,dia,tor}_ydisp` | Displacement Y component for each load. |
| `{ver,hor,dia,tor}_zdisp` | Displacement Z component for each load. |
| `{ver,hor,dia,tor}_stress` | Non-negative scalar stress for each load. Its physical unit is retained but not inferred. |

The training target per load is `[Ux, Uy, Uz, log1p(stress)]`. Stress must be
non-negative. Displacement components may be positive or negative.

## Changing a data path or file name

Edit only `configs/experiment.yaml`:

```yaml
dataset:
  metadata_file: data/raw/my_metadata.csv
  obj_directory: data/raw/meshes
  field_directory: data/raw/results
  obj_template: "part_{case_id}.obj"
  field_template: "results_{case_id}.csv"
```

Keep paths relative to this portable folder when possible. Absolute paths work
but reduce portability. Run `-Stage check-data` after every path change.

## Changing CSV header names without changing code

Map the canonical logical name to the header present in the new CSV:

```yaml
dataset:
  field_columns:
    surf: surface_flag
    x: node_x_mm
    y: node_y_mm
    z: node_z_mm
    ver_xdisp: vertical_ux
    ver_ydisp: vertical_uy
    ver_zdisp: vertical_uz
    ver_stress: vertical_von_mises_mpa
```

Do the same for every `hor_*`, `dia_*`, and `tor_*` field. The left-hand names
must not change; only update the right-hand CSV headers. This mapping changes
names, not physical meaning. If a new file changes coordinate frames, units,
stress definition, load meaning, or the interpretation of `surf`, it is a new
data contract and must be documented before mixing it with the old corpus.

## Splits and a new corpus

`data/splits/frozen_simjeb_split.yaml` is the exact 303-train / 37-validation
/ 38-test design split for the 378 paired SimJEB designs. Do not edit it to
reproduce the existing study. For a different corpus, create a new split file
with mutually exclusive design IDs, keep every load of a design in that one
split, and change `split_manifest` in `configs/experiment.yaml`. Fit field
normalization and select hyperparameters using train/validation only; reserve
test for final evaluation.
