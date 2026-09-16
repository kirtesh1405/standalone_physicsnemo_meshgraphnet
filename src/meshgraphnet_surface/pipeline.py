"""Manual preparation, training, and evaluation commands for the experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from .data import LOAD_CODES, TARGET_NAMES, SurfaceMesh, load_fields, load_obj, mesh_features, project_nearest, simplify, surface_targets, validate_ordered_surface
from .model import PhysicsNemoMeshGraphNet


@dataclass(frozen=True)
class Config:
    project_root: Path
    raw_root: Path
    metadata_file: Path
    obj_directory: Path
    field_directory: Path
    obj_template: str
    field_template: str
    column_map: dict[str, str]
    artifact_root: Path
    namespace: str
    split_manifest: Path
    seed: int
    target_face_count: int
    coordinate_tolerance: float
    hidden_width: int
    layers: int
    mlp_layers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    clip_norm: float
    critical_weight: float
    critical_quantile: float
    device: str
    require_cuda: bool
    amp: bool

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        config_path = Path(path).resolve()
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        root = config_path.parent.parent
        resolve = lambda value: (Path(value) if Path(value).is_absolute() else root / value).resolve()
        data, surface, model, training = values["dataset"], values["surface"], values["model"], values["training"]
        return cls(
            project_root=root, raw_root=resolve(data["raw_root"]), metadata_file=resolve(data["metadata_file"]),
            obj_directory=resolve(data["obj_directory"]), field_directory=resolve(data["field_directory"]),
            obj_template=str(data["obj_template"]), field_template=str(data["field_template"]), column_map=dict(data["field_columns"]),
            artifact_root=resolve(values["artifact_root"]), namespace=str(values["artifact_namespace"]), split_manifest=resolve(values["split_manifest"]),
            seed=int(values["seed"]), target_face_count=int(surface["target_face_count"]), coordinate_tolerance=float(surface["coordinate_tolerance"]),
            hidden_width=int(model["hidden_width"]), layers=int(model["message_passing_layers"]), mlp_layers=int(model["mlp_layers"]),
            epochs=int(training["epochs"]), learning_rate=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]),
            clip_norm=float(training["gradient_clip_norm"]), critical_weight=float(training["critical_weight"]), critical_quantile=float(training["critical_quantile"]),
            device=str(training["device"]), require_cuda=bool(training["require_cuda"]), amp=bool(training["amp"]),
        )

    @property
    def output_root(self) -> Path:
        return self.artifact_root / self.namespace

    @property
    def graph_root(self) -> Path:
        return self.output_root / "prepared_graphs"

    def source_paths(self, case_id: int) -> tuple[Path, Path]:
        values = {"case_id": int(case_id)}
        return self.obj_directory / self.obj_template.format(**values), self.field_directory / self.field_template.format(**values)


def _ids(config: Config, split: str) -> list[int]:
    values = yaml.safe_load(config.split_manifest.read_text(encoding="utf-8"))
    key = f"{split}_ids"
    if key not in values:
        raise ValueError(f"Split manifest does not contain '{key}'.")
    return [int(value) for value in values[key]]


def _all_ids(config: Config) -> list[int]:
    return [case_id for split in ("train", "validation", "test") for case_id in _ids(config, split)]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device(config: Config) -> torch.device:
    if config.device != "cuda":
        raise ValueError("This portable RTX 3050 configuration is deliberately CUDA-only.")
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable. Check docs/INSTALLATION.md; do not silently train on CPU.")
    return torch.device("cuda")


def check_data(config: Config) -> None:
    """Validate configured files and split IDs without generating graph artifacts."""
    errors: list[str] = []
    if not config.metadata_file.is_file():
        errors.append(f"Missing metadata file: {config.metadata_file}")
        metadata_ids: set[int] = set()
    else:
        metadata = pd.read_csv(config.metadata_file, usecols=["id"])
        metadata_ids = set(metadata["id"].astype(int))
    seen: set[int] = set()
    for split in ("train", "validation", "test"):
        for case_id in _ids(config, split):
            if case_id in seen:
                errors.append(f"Case {case_id} occurs in more than one split.")
            seen.add(case_id)
            if case_id not in metadata_ids:
                errors.append(f"Case {case_id} is in the split but absent from metadata.")
            obj, field = config.source_paths(case_id)
            if not obj.is_file(): errors.append(f"Case {case_id}: missing OBJ {obj}")
            if not field.is_file(): errors.append(f"Case {case_id}: missing field CSV {field}")
    if errors:
        raise FileNotFoundError("Data-contract check failed:\n- " + "\n- ".join(errors[:30]))
    print(f"Data check passed: {len(seen)} designs; {len(_ids(config, 'train'))} train, {len(_ids(config, 'validation'))} validation, {len(_ids(config, 'test'))} test.")


def prepare(config: Config, limit: int | None = None) -> None:
    """Create immutable, reduced PhysicsNeMo graph inputs from raw OBJ/CSV data."""
    _seed(config.seed)
    check_data(config)
    ids = _all_ids(config)
    if limit is not None:
        ids = ids[:int(limit)]
    config.graph_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for position, case_id in enumerate(ids, 1):
        destination = config.graph_root / f"{case_id}.npz"
        if destination.is_file():
            rows.append({"case_id": case_id, "state": "reused"})
            continue
        obj_path, field_path = config.source_paths(case_id)
        original = load_obj(obj_path)
        fields = load_fields(field_path, config.column_map)
        surface, alignment_error = validate_ordered_surface(original, fields, config.coordinate_tolerance)
        raw_targets = surface_targets(surface)
        reduced = simplify(original, config.target_face_count)
        reduced_targets = np.stack([project_nearest(original, values, reduced) for values in raw_targets])
        surface_codes = project_nearest(original, surface["surf"].to_numpy(np.float32)[:, None], reduced).reshape(-1).round().astype(np.uint8)
        node_features, edge_index, edge_attr, quality = mesh_features(reduced, surface_codes)
        reconstruction_index = cKDTree(reduced.vertices).query(original.vertices, k=1)[1].astype(np.int64)
        np.savez_compressed(destination, vertices=reduced.vertices, faces=reduced.faces, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, targets=reduced_targets, surface_codes=surface_codes, original_vertices=original.vertices, original_faces=original.faces, original_targets=raw_targets, reconstruction_index=reconstruction_index)
        rows.append({"case_id": case_id, "state": "prepared", "alignment_max_error": alignment_error, "original_vertices": len(original.vertices), "original_faces": len(original.faces), **quality})
        print(f"[{position}/{len(ids)}] prepared case {case_id}", flush=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.output_root / "prepare_manifest.csv", index=False)
    contract = {"split_manifest": str(config.split_manifest), "split_sha256": hashlib.sha256(config.split_manifest.read_bytes()).hexdigest(), "loads": list(LOAD_CODES), "targets": list(TARGET_NAMES), "target_transform": "Ux, Uy, Uz, log1p(stress)", "reduced_field_transfer": "nearest original vertex", "display_reconstruction": "nearest reduced vertex", "target_face_count": config.target_face_count}
    (config.output_root / "data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


def _normalization(config: Config, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for case_id in ids:
        with np.load(config.graph_root / f"{case_id}.npz", allow_pickle=False) as source:
            targets = np.asarray(source["targets"], dtype=np.float64)
            values.append(targets.reshape(-1, targets.shape[-1]))
    combined = np.concatenate(values)
    return combined.mean(axis=0).astype(np.float32), np.maximum(combined.std(axis=0), 1e-6).astype(np.float32)


def _graph(config: Config, case_id: int, load_index: int, mean: np.ndarray, std: np.ndarray, device: torch.device, original: bool = False, cached_edge: torch.Tensor | None = None) -> tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    with np.load(config.graph_root / f"{case_id}.npz", allow_pickle=False) as source:
        static = np.asarray(source["node_features"], dtype=np.float32)
        load = np.zeros((len(static), len(LOAD_CODES)), dtype=np.float32); load[:, load_index] = 1.0
        target_physical = np.asarray(source["targets"], dtype=np.float32)[load_index]
        edge = cached_edge if cached_edge is not None else torch.from_numpy(np.asarray(source["edge_attr"], dtype=np.float32)).to(device)
        graph = Data(edge_index=torch.from_numpy(np.asarray(source["edge_index"], dtype=np.int64)).to(device))
        extra = {"vertices": np.asarray(source["vertices"], dtype=np.float32)}
        if original:
            extra |= {"original_vertices": np.asarray(source["original_vertices"], dtype=np.float32), "original_faces": np.asarray(source["original_faces"], dtype=np.int32), "original_targets": np.asarray(source["original_targets"], dtype=np.float32), "reconstruction_index": np.asarray(source["reconstruction_index"], dtype=np.int64)}
    nodes = torch.from_numpy(np.concatenate((static, load), axis=1)).to(device)
    target = torch.from_numpy((target_physical - mean) / std).to(device)
    return graph, nodes, edge, target, {"physical": target_physical, **extra}


def _loss(prediction: torch.Tensor, target: torch.Tensor, config: Config) -> torch.Tensor:
    critical = target[:, 3] >= torch.quantile(target[:, 3], config.critical_quantile)
    weights = torch.ones(len(target), dtype=prediction.dtype, device=prediction.device)
    weights[critical] = config.critical_weight
    return ((prediction - target).abs().mean(dim=1) * weights).mean()


def _mean_loss(config: Config, model: torch.nn.Module, ids: list[int], mean: np.ndarray, std: np.ndarray, device: torch.device) -> float:
    model.eval(); values = []
    with torch.no_grad():
        for case_id in ids:
            for load_index in range(len(LOAD_CODES)):
                graph, nodes, edge, target, _ = _graph(config, case_id, load_index, mean, std, device)
                values.append(float(_loss(model(nodes, edge, graph), target, config).cpu()))
    return float(np.mean(values))


def train(config: Config, epochs: int | None = None) -> None:
    """Fit only on the training designs and monitor the frozen validation designs."""
    _seed(config.seed); device = _device(config)
    train_ids, validation_ids = _ids(config, "train"), _ids(config, "validation")
    mean, std = _normalization(config, train_ids)
    model = PhysicsNemoMeshGraphNet(hidden_dim=config.hidden_width, processor_size=config.layers, mlp_layers=config.mlp_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    epoch_count = epochs or config.epochs; history = []
    edge_cache: dict[int, torch.Tensor] = {}
    for epoch in range(1, epoch_count + 1):
        started = time.perf_counter(); model.train(); losses = []
        for case_id in np.random.default_rng(config.seed + epoch).permutation(train_ids):
            for load_index in range(len(LOAD_CODES)):
                case_id = int(case_id)
                graph, nodes, edge, target, _ = _graph(config, case_id, load_index, mean, std, device, cached_edge=edge_cache.get(case_id))
                edge_cache.setdefault(case_id, edge)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(nodes, edge, graph); loss = _loss(prediction, target, config)
                if not torch.isfinite(loss): raise FloatingPointError("Non-finite loss; no checkpoint was written.")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm); optimizer.step()
                losses.append(float(loss.detach().cpu()))
        validation_loss = _mean_loss(config, model, validation_ids, mean, std, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss, "epoch_seconds": time.perf_counter() - started}; history.append(row)
        print(f"epoch {epoch}/{epoch_count} | train {row['train_loss']:.6f} | validation {validation_loss:.6f} | {row['epoch_seconds']:.1f}s", flush=True)
    model_root = config.output_root / "models"; model_root.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "config": config.__dict__, "physicsnemo": True}, model_root / "meshgraphnet.pt")
    pd.DataFrame(history).to_csv(model_root / "history.csv", index=False)


def _metrics(actual: np.ndarray, predicted: np.ndarray, vertices: np.ndarray) -> dict[str, float]:
    def basic(first: np.ndarray, second: np.ndarray, name: str) -> dict[str, float]:
        residual = second - first
        return {f"{name}_mae": float(np.mean(np.abs(residual))), f"{name}_rmse": float(np.sqrt(np.mean(residual ** 2))), f"{name}_relative_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(first), 1e-12)), f"{name}_r2": float(1 - np.sum(residual ** 2) / max(np.sum((first - first.mean()) ** 2), 1e-12)), f"{name}_correlation": float(np.corrcoef(first, second)[0, 1]) if first.std() > 0 and second.std() > 0 else float("nan")}
    stress_actual, stress_predicted = np.expm1(actual[:, 3]), np.maximum(np.expm1(predicted[:, 3]), 0)
    disp_actual, disp_predicted = np.linalg.norm(actual[:, :3], axis=1), np.linalg.norm(predicted[:, :3], axis=1)
    result = basic(stress_actual, stress_predicted, "stress") | basic(disp_actual, disp_predicted, "displacement")
    for name, first, second in (("stress", stress_actual, stress_predicted), ("displacement", disp_actual, disp_predicted)):
        actual_peak, predicted_peak = int(np.argmax(first)), int(np.argmax(second))
        result[f"max_{name}_error_percent"] = float((second[predicted_peak] - first[actual_peak]) / max(first[actual_peak], 1e-12) * 100)
        result[f"{name}_hotspot_distance"] = float(np.linalg.norm(vertices[actual_peak] - vertices[predicted_peak]))
        for percent in (1, 5, 10):
            count = max(1, int(np.ceil(len(first) * percent / 100)))
            left, right = set(np.argpartition(first, -count)[-count:]), set(np.argpartition(second, -count)[-count:])
            result[f"{name}_top_{percent}_overlap"] = float(len(left & right) / count)
    return result


def evaluate(config: Config, split: str) -> None:
    """Save physical-space predictions and metrics for one named split."""
    device = _device(config); checkpoint_path = config.output_root / "models" / "meshgraphnet.pt"
    if not checkpoint_path.is_file(): raise FileNotFoundError("Missing model checkpoint. Run training before evaluation.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mean, std = np.asarray(checkpoint["mean"], dtype=np.float32), np.asarray(checkpoint["std"], dtype=np.float32)
    model = PhysicsNemoMeshGraphNet(hidden_dim=config.hidden_width, processor_size=config.layers, mlp_layers=config.mlp_layers).to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    prediction_root = config.output_root / "predictions" / split; prediction_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for case_id in _ids(config, split):
            for load_index, load_name in enumerate(LOAD_CODES):
                graph, nodes, edge, target, item = _graph(config, case_id, load_index, mean, std, device, original=True)
                torch.cuda.synchronize(); started = time.perf_counter(); normalized = model(nodes, edge, graph); torch.cuda.synchronize()
                physical = (normalized.float() * torch.from_numpy(std).to(device) + torch.from_numpy(mean).to(device)).cpu().numpy()
                actual = item["physical"]; metrics = _metrics(actual, physical, item["vertices"])
                rows.append({"split": split, "case_id": case_id, "load_case": load_name, "inference_seconds": time.perf_counter() - started, **metrics})
                mapping = item["reconstruction_index"]
                np.savez_compressed(prediction_root / f"{case_id}_{load_name}.npz", vertices=item["original_vertices"], faces=item["original_faces"], actual=item["original_targets"][load_index], predicted=physical[mapping])
    metrics_root = config.output_root / "metrics"; metrics_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_root / f"meshgraphnet_{split}.csv", index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--stage", required=True, choices=("check-data", "prepare", "train", "evaluate"))
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--epochs", type=int, help="Override epochs; use only for a deliberate smoke test.")
    parser.add_argument("--limit", type=int, help="Prepare only this many designs; use only for a smoke test.")
    args = parser.parse_args(argv); config = Config.load(args.config)
    if args.stage == "check-data": check_data(config)
    elif args.stage == "prepare": prepare(config, args.limit)
    elif args.stage == "train": train(config, args.epochs)
    else: evaluate(config, args.split)


if __name__ == "__main__":
    main()
