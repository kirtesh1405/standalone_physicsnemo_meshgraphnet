# Installation guide — Windows RTX 3050 Laptop GPU

## Supported profile

This experiment is pinned to the original hardware profile: NVIDIA GeForce RTX
3050 Laptop GPU with 4 GB VRAM, CUDA-enabled PyTorch 2.11.0 (`cu128`),
PhysicsNeMo 2.2.x, FP32, and one irregular graph at a time. Python 3.11 is the
recommended interpreter. The workflow intentionally stops if CUDA is not
available; it will not silently fall back to CPU.

Install a recent NVIDIA display driver that supports CUDA 12.8 before creating
the environment. The driver is sufficient; the full CUDA toolkit is not
required for the pip-based setup.

## Create the environment

Open PowerShell in this portable folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-pytorch-cu128.txt
python -m pip install -r requirements.txt
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
python -m pip install -e .
```

If the `py -3.11` launcher is unavailable, install Python 3.11 from the
official Python distribution, then repeat using its `python.exe` path.

`torch-scatter` is installed from the matching prebuilt PyG wheel page to avoid
a native Windows compiler build. Do not install PhysicsNeMo's optional `gnns`
group for this experiment: it can attempt an unsupported local build of extra
PyG extensions that this MeshGraphNet workflow does not use.

## Verify before preparing data

```powershell
.\.venv\Scripts\python.exe -c "import torch, physicsnemo; print('PhysicsNeMo:', physicsnemo.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected result: `CUDA: True` and `NVIDIA GeForce RTX 3050 Laptop GPU` (or the
same GPU model). If it shows `False`, stop and resolve the driver/PyTorch
installation before training.

Then check the raw-data location and headers without writing graph artifacts:

```powershell
.\scripts\run_experiment.ps1 -Stage check-data
```

## If a dependency installation fails

First confirm the environment is activated and that `python --version` is
3.11.x. Ensure the PyTorch CUDA requirements were installed before
`requirements.txt`. If the PyG wheel URL reports that no matching wheel exists,
use the matching PyTorch and CUDA selector from the official PyTorch Geometric
installation page; keep its versions aligned with `torch==2.11.0+cu128`.

Do not mix CPU-only PyTorch with the CUDA wheel in one environment.
