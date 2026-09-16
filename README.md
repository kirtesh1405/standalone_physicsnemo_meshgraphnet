# Portable PhysicsNeMo MeshGraphNet surface-field experiment

This folder is a standalone, manual workflow for training and reviewing a
PhysicsNeMo MeshGraphNet on load-conditioned surface displacement and stress
fields. It is safe to zip and move to another Windows machine. It does not
modify the parent project, raw source data, or its existing results.

## What is included

- Exact RTX 3050 experiment settings: 2,048 faces, width 48, six message-passing blocks, FP32, batch size one, 50 epochs.
- The exact 303 / 37 / 38 design-level SimJEB split used in the prior run.
- Raw-data validation, graph preparation, training, validation/test evaluation, and saved physical-space predictions.
- One Dash page for actual vs predicted vs absolute-error mesh views, scatter plot, case metrics, and learning curve.
- Manual setup and data-format instructions.

## Start here

1. Copy raw data into `data/raw/` following [the data contract](docs/DATA_CONTRACT.md).
2. Install the dedicated environment following [the installation guide](docs/INSTALLATION.md).
3. From this folder, run:

```powershell
.\scripts\run_experiment.ps1 -Stage all -Epochs 50
```

4. Review the saved test predictions:

```powershell
.\scripts\open_results_app.ps1
```

Open `http://127.0.0.1:8054` in a browser.

## Manual commands

The all-in-one script is simply a convenience wrapper. To run deliberately one
stage at a time, use:

```powershell
.\.venv\Scripts\python.exe -m meshgraphnet_surface.pipeline --stage check-data
.\.venv\Scripts\python.exe -m meshgraphnet_surface.pipeline --stage prepare
.\.venv\Scripts\python.exe -m meshgraphnet_surface.pipeline --stage train --epochs 50
.\.venv\Scripts\python.exe -m meshgraphnet_surface.pipeline --stage evaluate --split validation
.\.venv\Scripts\python.exe -m meshgraphnet_surface.pipeline --stage evaluate --split test
```

The test split is final-only. Do not repeatedly evaluate or tune against it.

## Important documents

- [Installation guide](docs/INSTALLATION.md) — Windows, CUDA, PhysicsNeMo, and verification.
- [Data contract](docs/DATA_CONTRACT.md) — directory layout, required OBJ/CSV contents, and how to map changed CSV headers.
- [Model and experiment specification](docs/MODEL_SPECIFICATION.md) — MeshGraphNet architecture, targets, loss, metrics, and limitations.
- [Operating guide](docs/OPERATING_GUIDE.md) — expected outputs, timing, reruns, and app use.

## Portable-folder layout

```text
standalone_physicsnemo_meshgraphnet/
├── configs/                 # edit experiment.yaml when paths or headers change
├── data/raw/                # user-supplied raw data; intentionally excluded from Git/zip
├── data/splits/             # frozen design split
├── artifacts/               # generated graphs, model, predictions, metrics
├── scripts/                 # manual PowerShell entry points
├── src/                     # all pipeline and app source code
└── docs/                    # operating instructions
```

Generated artifacts are deliberately separate from raw inputs. To make a
portable archive after a successful run, include `artifacts/`; to make a
small code-only archive, omit `data/raw/` and `artifacts/`.
