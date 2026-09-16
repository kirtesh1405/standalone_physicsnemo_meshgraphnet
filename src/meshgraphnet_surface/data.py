"""Input readers and deterministic surface-graph preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LOAD_CODES = ("ver", "hor", "dia", "tor")
TARGET_NAMES = ("ux", "uy", "uz", "log_stress")


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray


def load_obj(path: Path) -> SurfaceMesh:
    """Read OBJ vertices and fan-triangulate faces (OBJ indices are 1-based)."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("rt", encoding="utf-8", errors="strict") as source:
        for line_number, line in enumerate(source, 1):
            if line.startswith("v "):
                words = line.split()
                if len(words) < 4:
                    raise ValueError(f"Invalid OBJ vertex: {path}:{line_number}")
                vertices.append(tuple(map(float, words[1:4])))
            elif line.startswith("f "):
                indices = []
                for word in line.split()[1:]:
                    raw = int(word.split("/", 1)[0])
                    indices.append(raw - 1 if raw > 0 else len(vertices) + raw)
                if len(indices) < 3:
                    raise ValueError(f"Invalid OBJ face: {path}:{line_number}")
                faces.extend((indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1))
    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"{path} has no valid 3D vertices.")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
        raise ValueError(f"{path} has no valid triangular faces.")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError(f"{path} contains out-of-range face indices.")
    return SurfaceMesh(points, triangles)


def load_fields(path: Path, column_map: Mapping[str, str]) -> pd.DataFrame:
    """Read and rename a field CSV into the canonical experiment schema."""
    required = ("surf", "x", "y", "z") + tuple(
        f"{load}_{suffix}" for load in LOAD_CODES for suffix in ("xdisp", "ydisp", "zdisp", "stress")
    )
    missing_map = [name for name in required if name not in column_map]
    if missing_map:
        raise ValueError("field_columns is missing canonical keys: " + ", ".join(missing_map))
    header = pd.read_csv(path, nrows=0).columns
    missing_source = [column_map[name] for name in required if column_map[name] not in header]
    if missing_source:
        raise ValueError(f"{path.name} is missing configured CSV columns: " + ", ".join(missing_source))
    source_columns = [column_map[name] for name in required]
    frame = pd.read_csv(path, usecols=source_columns, dtype="float32")
    frame = frame.rename(columns={column_map[name]: name for name in required})
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path.name} contains non-finite values in required columns.")
    if (frame["surf"] < 0).any():
        raise ValueError(f"{path.name} contains negative surface labels.")
    return frame


def validate_ordered_surface(mesh: SurfaceMesh, fields: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, float]:
    """Require OBJ vertices and nonzero-``surf`` CSV rows to match in order."""
    surface = fields.loc[fields["surf"] != 0].reset_index(drop=True)
    coordinates = surface.loc[:, ["x", "y", "z"]].to_numpy(np.float64)
    if len(mesh.vertices) != len(coordinates):
        raise ValueError(
            "OBJ vertex count differs from CSV surface row count: "
            f"{len(mesh.vertices)} != {len(coordinates)}."
        )
    maximum = float(np.linalg.norm(mesh.vertices.astype(np.float64) - coordinates, axis=1).max())
    if maximum > tolerance:
        raise ValueError(f"OBJ/CSV ordered-coordinate error {maximum:.6g} exceeds tolerance {tolerance:.6g}.")
    return surface, maximum


def simplify(mesh: SurfaceMesh, face_count: int) -> SurfaceMesh:
    """Use deterministic QEM simplification without modifying the raw mesh."""
    import trimesh

    source = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False, validate=False)
    reduced = source.simplify_quadric_decimation(face_count=int(face_count), aggression=7)
    if len(reduced.faces) < 16 or len(reduced.vertices) < 16:
        raise ValueError("Mesh simplification produced an unusable surface.")
    return SurfaceMesh(np.asarray(reduced.vertices, dtype=np.float32), np.asarray(reduced.faces, dtype=np.int32))


def project_nearest(source: SurfaceMesh, values: np.ndarray, target: SurfaceMesh) -> np.ndarray:
    """Transfer values to a reduced mesh by nearest original source vertex."""
    indices = cKDTree(source.vertices.astype(np.float64)).query(target.vertices.astype(np.float64), k=1)[1]
    return np.asarray(values[indices], dtype=np.float32)


def mesh_features(mesh: SurfaceMesh, surface_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Derive PhysicsNeMo-Mesh topology, normals, and curvature features."""
    import torch
    from physicsnemo.mesh import Mesh

    native = Mesh(points=torch.from_numpy(mesh.vertices), cells=torch.from_numpy(mesh.faces.astype(np.int64)))
    normals = np.nan_to_num(native.point_normals.detach().cpu().numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    curvature = np.nan_to_num(native.mean_curvature_vertices.detach().cpu().numpy().astype(np.float32).reshape(-1, 1), nan=0.0, posinf=0.0, neginf=0.0)
    start, end = native.get_point_to_points_adjacency().expand_to_pairs()
    edge_index = torch.stack((start, end)).detach().cpu().numpy().astype(np.int64)
    center = mesh.vertices.mean(axis=0, keepdims=True)
    scale = max(float(np.linalg.norm(mesh.vertices - center, axis=1).max()), np.finfo(np.float32).eps)
    xyz = (mesh.vertices - center) / scale
    classes = np.eye(3, dtype=np.float32)[np.clip(surface_codes.astype(int) - 1, 0, 2)]
    node_features = np.concatenate((xyz.astype(np.float32), classes, normals, curvature / max(float(curvature.std()), 1e-6)), axis=1)
    edge_vectors = mesh.vertices[edge_index[1]] - mesh.vertices[edge_index[0]]
    edge_scale = max(float(np.linalg.norm(edge_vectors, axis=1).mean()), np.finfo(np.float32).eps)
    edge_features = np.concatenate((edge_vectors / edge_scale, np.linalg.norm(edge_vectors, axis=1, keepdims=True) / edge_scale), axis=1).astype(np.float32)
    quality = {
        "is_manifold": bool(native.is_manifold()), "is_watertight": bool(native.is_watertight()),
        "vertex_count": int(len(mesh.vertices)), "face_count": int(len(mesh.faces)),
    }
    return node_features, edge_index, edge_features, quality


def surface_targets(surface: pd.DataFrame) -> np.ndarray:
    """Build [load, original vertex, Ux/Uy/Uz/log1p(stress)] target tensors."""
    values = []
    for load in LOAD_CODES:
        stress = surface[f"{load}_stress"].to_numpy(np.float32)
        if (stress < 0).any():
            raise ValueError(f"{load}_stress has negative values; log1p target is undefined by this contract.")
        values.append(np.column_stack((
            surface[f"{load}_xdisp"], surface[f"{load}_ydisp"], surface[f"{load}_zdisp"], np.log1p(stress),
        )))
    return np.stack(values).astype(np.float32)
