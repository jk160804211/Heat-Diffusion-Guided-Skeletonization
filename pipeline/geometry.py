from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay, cKDTree


def compute_triangulation(points: np.ndarray, k: int = 10) -> np.ndarray:
    """Local PCA + 2D Delaunay triangulation, matching computeTriangulation.m."""
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n < 4:
        return np.zeros((0, 3), dtype=np.int64)

    k = int(max(3, min(k, n - 1)))
    tree = cKDTree(points)
    _, nbrs = tree.query(points, k=k + 1)
    faces: list[np.ndarray] = []

    for i in range(n):
        idx = np.asarray(nbrs[i, 1:], dtype=np.int64)
        local = points[idx]
        if local.shape[0] < 3:
            continue

        cov = np.cov(local, rowvar=False)
        try:
            eig_val, eig_vec = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        order = np.argsort(eig_val)
        tangent = eig_vec[:, order[1:3]]
        xy = (local - points[i]) @ tangent

        try:
            tri = Delaunay(xy, qhull_options="QJ")
        except Exception:
            continue
        faces.append(idx[tri.simplices])

    if not faces:
        return np.zeros((0, 3), dtype=np.int64)

    raw = np.vstack(faces).astype(np.int64)
    raw = raw[np.all(np.diff(np.sort(raw, axis=1), axis=1) > 0, axis=1)]
    if raw.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    key = np.sort(raw, axis=1)
    _, keep = np.unique(key, axis=0, return_index=True)
    return raw[np.sort(keep)]


def build_laplacian(
    points: np.ndarray, faces: np.ndarray
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Build a cotangent Laplacian and lumped mass matrix."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    n = points.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    mass = np.zeros(n, dtype=float)
    areas = np.zeros(faces.shape[0], dtype=float)

    def add(i: int, j: int, value: float) -> None:
        rows.append(i)
        cols.append(j)
        vals.append(value)

    for fi, (ia, ib, ic) in enumerate(faces):
        a, b, c = points[ia], points[ib], points[ic]
        nvec = np.cross(b - a, c - a)
        area2 = float(np.linalg.norm(nvec))
        if not np.isfinite(area2) or area2 < 1e-12:
            continue
        area = 0.5 * area2
        areas[fi] = area
        mass[[ia, ib, ic]] += area / 3.0

        cot_a = float(np.dot(b - a, c - a) / area2)
        cot_b = float(np.dot(a - b, c - b) / area2)
        cot_c = float(np.dot(a - c, b - c) / area2)

        for i, j, w in ((ib, ic, 0.5 * cot_a), (ia, ic, 0.5 * cot_b), (ia, ib, 0.5 * cot_c)):
            if not np.isfinite(w):
                continue
            add(i, i, w)
            add(j, j, w)
            add(i, j, -w)
            add(j, i, -w)

    eps = 1e-10
    add_values = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    lap = add_values + eps * sparse.eye(n, format="csr")
    mass[mass <= 0] = np.median(mass[mass > 0]) if np.any(mass > 0) else 1.0
    mass_mat = sparse.diags(mass, format="csr")
    return lap, mass_mat, areas


def heat_diffusion(
    points: np.ndarray,
    mass: sparse.csr_matrix,
    lap: sparse.csr_matrix,
    heat_time: float = 1000.0,
) -> np.ndarray:
    z_min = float(points[:, 2].min())
    source = (points[:, 2] >= z_min) & (points[:, 2] <= z_min + 0.2)
    b = np.zeros(points.shape[0], dtype=float)
    b[source] = 100.0
    rhs = mass @ b
    lhs = mass + heat_time * lap
    u = spsolve(lhs.tocsc(), rhs)
    return np.asarray(u, dtype=float)


def face_gradients(points: np.ndarray, faces: np.ndarray, scalar: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    scalar = np.asarray(scalar, dtype=float).reshape(-1)
    grad = np.zeros((faces.shape[0], 3), dtype=float)

    for fi, (ia, ib, ic) in enumerate(faces):
        a, b, c = points[ia], points[ib], points[ic]
        nvec = np.cross(b - a, c - a)
        area2 = float(np.linalg.norm(nvec))
        if area2 < 1e-12:
            continue
        nhat = nvec / area2
        ga = np.cross(nhat, c - b) / area2
        gb = np.cross(nhat, a - c) / area2
        gc = np.cross(nhat, b - a) / area2
        grad[fi] = scalar[ia] * ga + scalar[ib] * gb + scalar[ic] * gc
    return grad


def face_to_vertex_vectors(
    faces: np.ndarray, face_vectors: np.ndarray, n_vertices: int
) -> tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    face_vectors = np.asarray(face_vectors, dtype=float)
    sums = np.zeros((n_vertices, 3), dtype=float)
    counts = np.zeros(n_vertices, dtype=float)
    for fi, face in enumerate(faces):
        for vid in face:
            sums[vid] += face_vectors[fi]
            counts[vid] += 1.0
    counts[counts == 0] = 1.0
    vectors = sums / counts[:, None]
    mag = np.linalg.norm(vectors, axis=1) + 1e-10
    return vectors / mag[:, None], mag


def compute_growth_vectors(
    points: np.ndarray,
    k: int = 10,
    heat_time: float = 1000.0,
    progress: callable | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if progress:
        progress("局部三角化")
    faces = compute_triangulation(points, k)
    if faces.size == 0:
        raise RuntimeError("Triangulation produced no faces.")

    if progress:
        progress(f"构建 Laplacian ({faces.shape[0]} faces)")
    lap, mass, areas = build_laplacian(points, faces)

    if progress:
        progress("热扩散求解")
    heat = heat_diffusion(points, mass, lap, heat_time)

    if progress:
        progress("计算生长向量")
    fgrad = face_gradients(points, faces, heat)
    vectors, mag = face_to_vertex_vectors(faces, fgrad, points.shape[0])
    return vectors, {"faces": faces, "heat": heat, "growth_magnitude": mag, "face_area": areas}

