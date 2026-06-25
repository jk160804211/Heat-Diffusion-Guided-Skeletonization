from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from scipy.io import loadmat


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load a point cloud and return an Nx3 float array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".mat":
        mat = loadmat(path)
        for key in ("data", "points", "ptcloud", "pointCloud", "xyz"):
            if key in mat:
                arr = np.asarray(mat[key], dtype=float)
                break
        else:
            candidates = [
                np.asarray(v, dtype=float)
                for k, v in mat.items()
                if not k.startswith("__") and np.asarray(v).ndim == 2
            ]
            if not candidates:
                raise ValueError(f"No Nx3 matrix found in {path}")
            arr = max(candidates, key=lambda a: a.shape[0])
    else:
        arr = np.loadtxt(path, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected an Nx3 or wider point matrix, got {arr.shape}")

    xyz = np.asarray(arr[:, :3], dtype=float)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if xyz.size == 0:
        raise ValueError(f"No finite XYZ points found in {path}")
    return xyz


def save_csv(path: str | Path, data: np.ndarray, header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data)
    fmt = "%d" if np.issubdtype(arr.dtype, np.integer) else "%.10f"
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt=fmt)


def save_obj_lines(path: str | Path, nodes: np.ndarray, edges: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# skeleton line object\n")
        for p in nodes:
            fh.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        for u, v in edges:
            fh.write(f"l {u + 1} {v + 1}\n")


def save_dict_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            vals = []
            for key in headers:
                val = row.get(key, "")
                if isinstance(val, float):
                    vals.append(f"{val:.10f}")
                else:
                    vals.append(str(val))
            fh.write(",".join(vals) + "\n")


def save_json(path: str | Path, data: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
