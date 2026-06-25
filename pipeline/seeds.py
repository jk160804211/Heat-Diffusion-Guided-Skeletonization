from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def assign_labels(points: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    if seeds.size == 0:
        return np.zeros(points.shape[0], dtype=np.int64)
    tree = cKDTree(np.asarray(seeds, dtype=float))
    _, labels = tree.query(np.asarray(points, dtype=float), k=1)
    return labels.astype(np.int64)


def farthest_point_sampling(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    count = int(min(max(count, 1), n))
    selected = np.zeros(count, dtype=np.int64)
    available = np.ones(n, dtype=bool)
    distances = np.full(n, np.inf, dtype=float)

    selected[0] = int(np.argmin(points[:, 2]))
    available[selected[0]] = False
    distances = np.linalg.norm(points - points[selected[0]], axis=1)

    for i in range(1, count):
        masked = np.where(available, distances, -np.inf)
        idx = int(np.argmax(masked))
        selected[i] = idx
        available[idx] = False
        distances = np.minimum(distances, np.linalg.norm(points - points[idx], axis=1))
    return selected


def estimate_r_knn(points: np.ndarray, k: int = 5, scale: float = 1.5, q: float = 0.5) -> float:
    points = np.asarray(points, dtype=float)
    k = int(max(1, min(k, points.shape[0] - 1)))
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=k + 1)
    base = float(np.quantile(dist[:, k], q))
    return max(scale * base, 1e-9)


def init_seeds_with_root(points: np.ndarray, init_count: int) -> np.ndarray:
    idx = farthest_point_sampling(points, init_count)
    seeds = points[idx].copy()
    root_mask = seeds[:, 2] < 0.5
    if not np.any(root_mask):
        root_mask[np.argmin(seeds[:, 2])] = True
    root = seeds[root_mask].mean(axis=0)
    seeds = seeds[~root_mask]
    return np.vstack([seeds, root])


def radius_merge(points: np.ndarray, radius: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    merged: list[np.ndarray] = []
    remaining = points.copy()
    while remaining.size:
        ref = remaining[0]
        dist = np.linalg.norm(remaining - ref, axis=1)
        close = dist < radius
        merged.append(remaining[close].mean(axis=0))
        remaining = remaining[~close]
    return np.vstack(merged) if merged else np.zeros((0, 3), dtype=float)


def set_seed_points_basic(
    points: np.ndarray,
    init_count: int,
    grid_step: float,
    progress: callable | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Python port of main3.m -> setSeedPoints(data, 450, gridStep)."""
    points = np.asarray(points, dtype=float)
    if progress:
        progress(f"main3 普通种子 FPS: {init_count}")
    selected = farthest_point_sampling(points, init_count)
    seeds = points[selected].copy()

    min_z = float(seeds[:, 2].min())
    max_z = float(seeds[:, 2].max())
    height = max_z - min_z    

    root_mask = seeds[:, 2] < min_z + height / 15.0
    if np.count_nonzero(root_mask) > 1:
        root = seeds[root_mask].mean(axis=0, keepdims=True)
    elif np.count_nonzero(root_mask) == 1:
        root = seeds[root_mask].copy()
    else:
        root = seeds[[int(np.argmin(seeds[:, 2]))]].copy()
        root_mask[int(np.argmin(seeds[:, 2]))] = True

    seeds = np.vstack([seeds[~root_mask], root])
    labels = assign_labels(points, seeds)

    centered: list[np.ndarray] = []
    for i in range(seeds.shape[0]):
        idx = np.flatnonzero(labels == i)
        if idx.size < 5:
            continue
        centered.append(points[idx].mean(axis=0))
    if not centered:
        centered = [points.mean(axis=0)]
    seeds_c = np.vstack(centered)

    merge_dist = grid_step * 5.0
    if progress:
        progress(f"main3 种子合并半径: {merge_dist:.4f}")
    seeds_c = radius_merge(seeds_c, merge_dist)
    labels_c = assign_labels(points, seeds_c)
    return seeds_c, labels_c


def geometric_median(points: np.ndarray, max_iter: int = 50) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros(3, dtype=float)
    m = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - m, axis=1)
        d[d < 1e-9] = 1e-9
        w = 1.0 / d
        nxt = np.sum(points * w[:, None], axis=0) / np.sum(w)
        if np.linalg.norm(nxt - m) < 1e-5:
            return nxt
        m = nxt
    return m


def plane_basis(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ref, axis))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u = u / max(np.linalg.norm(u), 1e-12)
    v = np.cross(axis, u)
    v = v / max(np.linalg.norm(v), 1e-12)
    return np.column_stack([u, v])


def circle_from_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> tuple[bool, np.ndarray, float]:
    mat = np.vstack([2.0 * (p2 - p1), 2.0 * (p3 - p1)])
    rhs = np.array([np.sum(p2 * p2 - p1 * p1), np.sum(p3 * p3 - p1 * p1)])
    if np.linalg.matrix_rank(mat) < 2:
        return False, np.zeros(2), 0.0
    center = np.linalg.solve(mat, rhs)
    radius = float(np.linalg.norm(p1 - center))
    ok = np.isfinite(radius) and radius > 0
    return bool(ok), center, radius


def circle_ls(points_2d: np.ndarray) -> np.ndarray:
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    mat = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    rhs = x * x + y * y
    sol, *_ = np.linalg.lstsq(mat, rhs, rcond=None)
    return sol[:2]


def ransac_circle_2d(
    points_2d: np.ndarray,
    rng: np.random.Generator,
    iters: int,
    inlier_tol: float,
    min_inliers: int,
) -> tuple[bool, np.ndarray]:
    m = points_2d.shape[0]
    if m < 3:
        return False, np.zeros(2)
    best: np.ndarray | None = None
    best_center: np.ndarray | None = None
    for _ in range(iters):
        ids = rng.choice(m, size=3, replace=False)
        ok, center, radius = circle_from_3points(points_2d[ids[0]], points_2d[ids[1]], points_2d[ids[2]])
        if not ok:
            continue
        err = np.abs(np.linalg.norm(points_2d - center, axis=1) - radius)
        inliers = np.flatnonzero(err < inlier_tol)
        if best is None or inliers.size > best.size:
            best = inliers
            best_center = center
    if best is not None and best.size >= max(min_inliers, 3):
        return True, circle_ls(points_2d[best])
    if best_center is not None:
        return False, best_center
    return False, np.zeros(2)


def set_seed_points_centered(
    points: np.ndarray,
    growth_vectors: np.ndarray,
    init_count: int,
    grid_step: float,
    max_iters: int = 3,
    tol_shift: float = 1e-3,
    min_pts: int = 12,
    merge_scale: float = 0.8,
    ransac_tol_scale: float = 0.015,
    inlier_frac: float = 0.5,
    random_seed: int = 18,
    progress: callable | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float] | int]]:
    points = np.asarray(points, dtype=float)
    growth_vectors = np.asarray(growth_vectors, dtype=float)
    rng = np.random.default_rng(random_seed)
    radius = estimate_r_knn(points, 5, 1.5, 0.5)
    merge_dist = max(grid_step * 6.0, merge_scale * radius)
    ransac_tol = max(1e-9, ransac_tol_scale * radius)

    seeds = init_seeds_with_root(points, init_count)
    info: dict[str, list[float] | int] = {"shifts": [], "num_centers": []}

    for iteration in range(max_iters):
        if progress:
            progress(f"种子中心校正 {iteration + 1}/{max_iters}")
        labels = assign_labels(points, seeds)
        new_seeds = np.zeros_like(seeds)

        for i in range(seeds.shape[0]):
            idx = np.flatnonzero(labels == i)
            if idx.size < min_pts:
                new_seeds[i] = geometric_median(points[idx]) if idx.size else seeds[i]
                continue

            nbr = points[idx]
            p0 = nbr.mean(axis=0)
            dist = np.linalg.norm(nbr - p0, axis=1)
            sigma = max(float(np.median(dist)) + np.finfo(float).eps, 1e-9)
            weights = np.exp(-((dist / sigma) ** 2))
            g = growth_vectors[idx]
            g_norm = np.linalg.norm(g, axis=1)
            g = g / np.maximum(g_norm[:, None], 1e-12)
            axis = weights @ g
            if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-6:
                try:
                    eig_val, eig_vec = np.linalg.eigh(np.cov(nbr, rowvar=False))
                    axis = eig_vec[:, int(np.argmax(eig_val))]
                except np.linalg.LinAlgError:
                    new_seeds[i] = geometric_median(nbr)
                    continue
            axis = axis / max(np.linalg.norm(axis), 1e-12)

            axial = (nbr - p0) @ axis
            proj = p0 + np.outer(axial, axis)
            radial = np.linalg.norm(nbr - proj, axis=1)
            rad_med = float(np.median(radial))
            rad_mad = 1.4826 * float(np.median(np.abs(radial - rad_med)))
            keep = radial < (rad_med + 2.5 * max(rad_mad, 1e-6))
            nbr = nbr[keep]
            if nbr.shape[0] < 6:
                new_seeds[i] = geometric_median(nbr)
                continue

            basis = plane_basis(axis)
            uv = (nbr - p0) @ basis
            min_inliers = int(np.ceil(inlier_frac * uv.shape[0]))
            ok, center_2d = ransac_circle_2d(uv, rng, 200, ransac_tol, min_inliers)
            new_seeds[i] = p0 + basis @ center_2d if ok else geometric_median(nbr)

        shift = float(np.mean(np.linalg.norm(new_seeds - seeds, axis=1) ** 2))
        info["shifts"].append(shift)  # type: ignore[index]
        seeds = new_seeds
        info["num_centers"].append(int(seeds.shape[0]))  # type: ignore[index]
        if shift < tol_shift:
            break

    # Root merge, matching the centered MATLAB function.
    height = float(seeds[:, 2].max() - seeds[:, 2].min())
    root_mask = seeds[:, 2] <= seeds[:, 2].min() + height / 15.0
    if not np.any(root_mask):
        root_mask[np.argmin(seeds[:, 2])] = True
    root = seeds[root_mask].mean(axis=0)
    seeds = np.vstack([seeds[~root_mask], root])

    # Keep this helper available for future use; final MATLAB version has it disabled.
    _ = merge_dist
    labels = assign_labels(points, seeds)
    info["final_centers"] = int(seeds.shape[0])
    return seeds, labels, info
