# PhysicsNeMo MeshGraphNet model specification

## Experiment question

Can a mesh-connectivity model predict a selected load case's surface
displacement components and stress from the reduced OBJ geometry and load
identity, without reading a solver result at inference time?

This is a surface-field surrogate experiment. It is not a volumetric FEM
surrogate and is not an engineering acceptance or safety calculation.

## Graph and features

Each raw OBJ surface is quadratically simplified to a target of **2,048
triangles**. PhysicsNeMo-Mesh then derives directed point-to-point adjacency,
native vertex normals, mean vertex curvature, and topology checks.

Each vertex receives 14 input features:

| Feature group | Dimensions | Description |
|---|---:|---|
| Centered/scaled position | 3 | XYZ centered per design and scaled by the farthest vertex distance. |
| Surface class | 3 | One-hot version of the source `surf` label, clipped to classes 1–3. |
| Normal | 3 | Native PhysicsNeMo-Mesh point normal. |
| Curvature | 1 | Native mean curvature, normalized within the reduced mesh. |
| Load identity | 4 | One-hot `ver`, `hor`, `dia`, or `tor`, broadcast to vertices. |

Each directed edge has four features: normalized relative `dx, dy, dz` and
normalized edge length.

## Network and training settings

| Setting | Value |
|---|---:|
| Framework | NVIDIA PhysicsNeMo MeshGraphNet |
| Hidden width | 48 |
| Message-passing blocks | 6 |
| Node/edge processor MLP layers | 2 |
| Output channels | 4 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Gradient clipping | L2 norm 1.0 |
| Precision | FP32 (`amp: false`) |
| Batch size | 1 graph |
| Epochs | 50 |
| Seed | 42 |

The outputs are normalized `[Ux, Uy, Uz, log1p(stress)]` values. Mean and
standard deviation are fit using training designs only and stored in the model
checkpoint. At evaluation, stress is restored as `max(expm1(prediction), 0)`
and displacement magnitude is `sqrt(Ux² + Uy² + Uz²)`.

The optimization loss is mean absolute error over the four normalized target
channels. Vertices at or above the per-graph 90th percentile of normalized
log-stress receive a weight of 2.0. This encourages critical-region learning,
but is not a physical loss or a replacement for FEM validation.

## Split and leakage protection

The supplied SimJEB split has 378 paired designs: 303 train, 37 validation,
and 38 test. Every design's four loads remain in one split. Normalization and
weights are fit only on train; validation monitors learning; test is only for
final reporting.

## Evaluation outputs

For each design/load pair the pipeline reports stress and displacement
magnitude MAE, RMSE, relative L2, R², correlation, maximum-field error,
hotspot distance, and top 1/5/10% critical-region overlap. Saved prediction
files contain original OBJ vertices/faces plus actual and predicted four-channel
fields. The app reconstructs every original OBJ vertex from its nearest reduced
vertex; this is a visualization/evaluation approximation, not barycentric
interpolation.

## Known limitations

- Stress unit and stress measure are not inferred from source CSV headers.
- Load magnitude, material properties, and boundary conditions are represented only by four load identities in the supplied experiment.
- QEM reduction and nearest-vertex field transfer can affect peaks.
- The model does not predict volumetric fields.
- Use a new design-level split for a fundamentally changed dataset or physical loading convention.
