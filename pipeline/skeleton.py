from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.special import gammaln
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

from .seeds import assign_labels


@dataclass(slots=True)
class SkeletonResult:
    stream_points: np.ndarray
    raw_edges: np.ndarray
    refined_points: np.ndarray
    refined_edges: np.ndarray
    smooth_points: np.ndarray
    root_to_leaf: list[list[int]]


class LocalVectorField:
    def __init__(self, points: np.ndarray, vectors: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.vectors = np.asarray(vectors, dtype=float)
        self.tree = cKDTree(self.points)
        self.bounds_min = self.points.min(axis=0)
        self.bounds_max = self.points.max(axis=0)
        if self.points.shape[0] > 2:
            d2, _ = self.tree.query(self.points, k=min(2, self.points.shape[0]))
            nn = d2[:, -1]
            self.support_radius = max(0.6, 6.0 * float(np.median(nn)))
        else:
            self.support_radius = 0.6
        self.linear: tuple[LinearNDInterpolator, LinearNDInterpolator, LinearNDInterpolator] | None = None
        span = self.bounds_max - self.bounds_min
        if self.points.shape[0] >= 8 and np.count_nonzero(span > 1e-8) == 3:
            try:
                self.linear = (
                    LinearNDInterpolator(self.points, self.vectors[:, 0], fill_value=np.nan),
                    LinearNDInterpolator(self.points, self.vectors[:, 1], fill_value=np.nan),
                    LinearNDInterpolator(self.points, self.vectors[:, 2], fill_value=np.nan),
                )
            except Exception:
                self.linear = None

    def vector_at(self, point: np.ndarray) -> np.ndarray | None:
        if self.linear is not None:
            vals = np.array([interp(point) for interp in self.linear], dtype=float).reshape(3)
            if np.all(np.isfinite(vals)) and np.linalg.norm(vals) > 1e-12:
                return vals
        if not self.inside(point, 0.0):
            return None
        _, idx = self.tree.query(point, k=1)
        vec = self.vectors[int(idx)]
        if not np.all(np.isfinite(vec)) or np.linalg.norm(vec) <= 1e-12:
            return None
        return vec

    def inside(self, point: np.ndarray, pad: float) -> bool:
        return bool(np.all(point >= self.bounds_min - pad) and np.all(point <= self.bounds_max + pad))

    def near_support(self, point: np.ndarray) -> bool:
        dist, _ = self.tree.query(point, k=1)
        return bool(np.isfinite(dist) and dist <= self.support_radius)


def _trace_one_direction(
    field: LocalVectorField,
    start: np.ndarray,
    sign: float,
    step_len: float,
    steps: int,
) -> list[np.ndarray]:
    current = np.asarray(start, dtype=float).copy()
    out: list[np.ndarray] = []
    pad = max(0.4 * step_len, 1e-9)
    for _ in range(steps):
        vec = field.vector_at(current)
        if vec is None:
            break
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm < 1e-12:
            break
        nxt = current + sign * step_len * vec / norm
        if not np.all(np.isfinite(nxt)) or not field.inside(nxt, pad) or not field.near_support(nxt):
            break
        current = nxt
        out.append(current.copy())
    return out


def stream_line_tracing(
    points: np.ndarray,
    seeds: np.ndarray,
    labels: np.ndarray,
    growth_vectors: np.ndarray,
    grid_resolution: tuple[int, int, int] = (15, 15, 15),
    step_len: float = 0.5,
    steps: int = 20,
    progress: callable | None = None,
) -> np.ndarray:
    """Trace candidate skeleton points from corrected seed points."""
    points = np.asarray(points, dtype=float)
    seeds = np.asarray(seeds, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    growth_vectors = np.asarray(growth_vectors, dtype=float)
    global_tree = cKDTree(points)
    stream_points: list[np.ndarray] = []

    for i, seed in enumerate(seeds):
        if progress and (i % 10 == 0 or i == seeds.shape[0] - 1):
            progress(f"流线追踪 {i + 1}/{seeds.shape[0]}")
        idx = np.flatnonzero(labels == i)
        if idx.size < 5:
            _, idx = global_tree.query(seed, k=min(10, points.shape[0]))
            idx = np.asarray(idx, dtype=np.int64).reshape(-1)

        field = LocalVectorField(points[idx], growth_vectors[idx])
        span = np.ptp(points[idx], axis=0)
        denom = np.maximum(np.asarray(grid_resolution, dtype=float) - 1.0, 1.0)
        cell = span / denom
        valid_cell = cell[np.isfinite(cell) & (cell > 1e-9)]
        local_step = step_len * float(np.mean(valid_cell)) if valid_cell.size else step_len
        local_step = max(local_step, 1e-4)
        back = _trace_one_direction(field, seed, -1.0, local_step, steps)
        front = _trace_one_direction(field, seed, 1.0, local_step, steps)
        path = list(reversed(back)) + [seed.copy()] + front
        if len(path) >= 2:
            label_col = np.full((len(path), 1), i, dtype=float)
            stream_points.append(np.column_stack([np.vstack(path), label_col]))

    if not stream_points:
        return np.zeros((0, 4), dtype=float)
    stream = np.vstack(stream_points)
    stream[:, :3] = np.clip(stream[:, :3], points.min(axis=0), points.max(axis=0))
    return stream


def build_knn_mst(points: np.ndarray, k: int = 12) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)

    k = int(max(1, min(k, n - 1)))
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=k + 1)

    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    vals = dist[:, 1:].reshape(-1)
    graph = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
    graph = graph.minimum(graph.T).tocsr()

    n_comp, comp = connected_components(graph, directed=False)
    if n_comp > 1:
        extra_rows: list[int] = []
        extra_cols: list[int] = []
        extra_vals: list[float] = []
        base = np.flatnonzero(comp == 0)
        for comp_id in range(1, n_comp):
            other = np.flatnonzero(comp == comp_id)
            base_tree = cKDTree(points[base])
            d, nearest = base_tree.query(points[other], k=1)
            j = int(np.argmin(d))
            u = int(other[j])
            v = int(base[int(nearest[j])])
            w = float(d[j])
            extra_rows.extend([u, v])
            extra_cols.extend([v, u])
            extra_vals.extend([w, w])
            base = np.concatenate([base, other])
        extra = sparse.coo_matrix((extra_vals, (extra_rows, extra_cols)), shape=(n, n)).tocsr()
        graph = (graph + extra).tocsr()

    mst = minimum_spanning_tree(graph).tocoo()
    edges = np.column_stack([mst.row, mst.col]).astype(np.int64)
    return edges


def build_complete_mst(points: np.ndarray, max_dense_nodes: int = 3500) -> np.ndarray:
    """MATLAB intialGraphGenerate equivalent: MST on the full distance graph."""
    points = np.asarray(points, dtype=float)
    n = points.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)
    if n > max_dense_nodes:
        return build_knn_mst(points, k=24)

    dist = squareform(pdist(points[:, :3], metric="euclidean"))
    np.fill_diagonal(dist, 0.0)
    mst = minimum_spanning_tree(dist).tocoo()
    return np.column_stack([mst.row, mst.col]).astype(np.int64)


def _estimate_spacing(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 1.0
    tree = cKDTree(points)
    dist, _ = tree.query(points, k=2)
    return max(float(np.median(dist[:, 1])), 1e-6)


def _edge_has_point_support(
    p0: np.ndarray,
    p1: np.ndarray,
    support_tree: cKDTree,
    support_radius: float,
) -> bool:
    samples = np.linspace(0.15, 0.85, 4)
    seg = p0[None, :] * (1.0 - samples[:, None]) + p1[None, :] * samples[:, None]
    dist, _ = support_tree.query(seg, k=1)
    return bool(np.all(dist <= support_radius))


def _add_weighted_edge(
    rows: list[int],
    cols: list[int],
    vals: list[float],
    u: int,
    v: int,
    w: float,
) -> None:
    if u == v or not np.isfinite(w) or w <= 0:
        return
    rows.extend([int(u), int(v)])
    cols.extend([int(v), int(u)])
    vals.extend([float(w), float(w)])


def build_topology_aware_mst(
    stream_points: np.ndarray,
    support_points: np.ndarray,
    neighbor_k: int = 40,
    connector_scale: float = 18.0,
    support_scale: float = 6.0,
    upward_penalty: float = 4.0,
) -> np.ndarray:
    """Build a tree using stream continuity plus local branch-base connectors."""
    xyz = np.asarray(stream_points[:, :3], dtype=float)
    labels = np.asarray(stream_points[:, 3], dtype=int)
    n = xyz.shape[0]
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    stream_tree = cKDTree(xyz)
    support_tree = cKDTree(support_points)
    spacing = _estimate_spacing(support_points)
    support_radius = max(0.35, support_scale * spacing)
    z_tol = max(0.25, 3.0 * spacing)
    max_connector = max(1.5, connector_scale * spacing)

    for lab in np.unique(labels):
        ids = np.flatnonzero(labels == lab)
        for u, v in zip(ids[:-1], ids[1:]):
            d = float(np.linalg.norm(xyz[u] - xyz[v]))
            _add_weighted_edge(rows, cols, vals, int(u), int(v), max(1e-6, 0.02 * d))

    for lab in np.unique(labels):
        ids = np.flatnonzero(labels == lab)
        if ids.size == 0:
            continue
        z = xyz[ids, 2]
        z_cut = np.quantile(z, 0.22)
        src_ids = ids[z <= z_cut]
        if src_ids.size == 0:
            src_ids = ids[[int(np.argmin(z))]]

        added = 0
        for src in src_ids:
            k = min(neighbor_k, n)
            dist, nbr = stream_tree.query(xyz[src], k=k)
            dist = np.asarray(dist).reshape(-1)
            nbr = np.asarray(nbr, dtype=int).reshape(-1)
            for d, dst in zip(dist, nbr):
                if dst == src or labels[dst] == lab:
                    continue
                if d > max_connector:
                    continue
                dz = xyz[dst, 2] - xyz[src, 2]
                if dz > z_tol:
                    continue
                if not _edge_has_point_support(xyz[src], xyz[dst], support_tree, support_radius):
                    continue
                upward_ratio = max(dz, 0.0) / max(float(d), 1e-6)
                weight = float(d) * (1.0 + upward_penalty * upward_ratio) + 0.15 * abs(float(dz))
                _add_weighted_edge(rows, cols, vals, int(src), int(dst), weight)
                added += 1
                if added >= 4:
                    break
            if added >= 4:
                break

    graph = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    n_comp, comp = connected_components(graph, directed=False)
    while n_comp > 1:
        best: tuple[float, int, int] | None = None
        for comp_id in range(n_comp):
            a = np.flatnonzero(comp == comp_id)
            b = np.flatnonzero(comp != comp_id)
            if a.size == 0 or b.size == 0:
                continue
            tree_b = cKDTree(xyz[b])
            dist, idx = tree_b.query(xyz[a], k=1)
            j = int(np.argmin(dist))
            u = int(a[j])
            v = int(b[int(idx[j])])
            dz = xyz[v, 2] - xyz[u, 2]
            penalty = 1.0 + upward_penalty * max(float(dz), 0.0) / max(float(dist[j]), 1e-6)
            cand = (float(dist[j]) * penalty, u, v)
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            break
        extra = sparse.coo_matrix(
            ([best[0], best[0]], ([best[1], best[2]], [best[2], best[1]])),
            shape=(n, n),
        ).tocsr()
        graph = graph + extra
        n_comp, comp = connected_components(graph, directed=False)

    mst = minimum_spanning_tree(graph).tocoo()
    return np.column_stack([mst.row, mst.col]).astype(np.int64)


def adjacency_from_edges(n_nodes: int, edges: np.ndarray) -> list[list[int]]:
    adj = [[] for _ in range(n_nodes)]
    for u, v in np.asarray(edges, dtype=np.int64):
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def root_to_leaf_paths(nodes: np.ndarray, edges: np.ndarray) -> list[list[int]]:
    n = nodes.shape[0]
    if n == 0:
        return []
    root = int(np.argmin(nodes[:, 2]))
    adj = adjacency_from_edges(n, edges)
    parent = np.full(n, -1, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    stack = [root]
    seen[root] = True
    order = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                stack.append(v)
                order.append(v)

    leaves = [i for i in order if i != root and len(adj[i]) <= 1]
    if not leaves and n > 1:
        leaves = [order[-1]]
    paths: list[list[int]] = []
    for leaf in leaves:
        cur = leaf
        path: list[int] = []
        while cur >= 0:
            path.append(int(cur))
            if cur == root:
                break
            cur = int(parent[cur])
        paths.append(list(reversed(path)))
    return paths


def _compact_to_largest_component(nodes: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = nodes.shape[0]
    if n == 0 or edges.size == 0:
        return nodes, edges.reshape(0, 2)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    vals = np.ones(rows.size)
    mat = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    n_comp, comp = connected_components(mat, directed=False)
    if n_comp <= 1:
        return nodes, edges
    sizes = np.bincount(comp)
    keep_comp = int(np.argmax(sizes))
    keep = np.flatnonzero(comp == keep_comp)
    remap = np.full(n, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size)
    mask = (remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)
    new_edges = remap[edges[mask]]
    return nodes[keep], np.asarray(new_edges, dtype=np.int64)


def redundant_filter(
    skeleton_points: np.ndarray, edges: np.ndarray, filter_num: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    if skeleton_points.shape[0] < 2 or edges.size == 0:
        return skeleton_points, edges.reshape(0, 2)
    paths = root_to_leaf_paths(skeleton_points[:, :3], edges)
    if not paths:
        return skeleton_points, edges

    drop: set[int] = set()
    path_sets = [set(path) for path in paths]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            diff = len(path_sets[i] - path_sets[j])
            if diff < filter_num:
                drop.add(j if len(paths[i]) > len(paths[j]) else i)

    kept = [p for idx, p in enumerate(paths) if idx not in drop]
    if not kept:
        kept = [max(paths, key=len)]

    edge_set: set[tuple[int, int]] = set()
    for path in kept:
        for a, b in zip(path[:-1], path[1:]):
            edge_set.add(tuple(sorted((a, b))))
    if not edge_set:
        return skeleton_points, edges.reshape(0, 2)
    new_edges = np.array(sorted(edge_set), dtype=np.int64)
    return _compact_to_largest_component(skeleton_points, new_edges)


def growth_trend_filter(
    skeleton_points: np.ndarray,
    edges: np.ndarray,
    filter_angle: float = np.pi / 2,
    filter_angle2: float = np.pi / 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative port of growthTrendFilter.

    The MATLAB version reconnects invalid path segments with repeated nearest
    neighbor searches. For the interactive first version we keep the topology
    stable and remove only isolated non-finite nodes.
    """
    _ = filter_angle, filter_angle2
    finite = np.all(np.isfinite(skeleton_points[:, :3]), axis=1)
    if np.all(finite):
        return skeleton_points, edges
    remap = np.full(skeleton_points.shape[0], -1, dtype=np.int64)
    keep = np.flatnonzero(finite)
    remap[keep] = np.arange(keep.size)
    mask = (remap[edges[:, 0]] >= 0) & (remap[edges[:, 1]] >= 0)
    return skeleton_points[keep], remap[edges[mask]]


def bezier_curve(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    m = points.shape[0]
    if m <= 2:
        return points.copy()
    degree = m - 1
    t = np.linspace(1e-10, 1.0 - 1e-10, m)
    out = np.zeros_like(points)
    for i in range(m):
        log_bin = gammaln(degree + 1) - gammaln(i + 1) - gammaln(degree - i + 1)
        coeff = np.exp(log_bin + (degree - i) * np.log1p(-t) + i * np.log(t))
        out += coeff[:, None] * points[i]
    return out


def bezier_smooth(
    skeleton_points: np.ndarray, edges: np.ndarray, filter_num: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    skeleton_points, edges = redundant_filter(skeleton_points, edges, filter_num)
    xyz = skeleton_points[:, :3]
    smooth = np.zeros((skeleton_points.shape[0], 4), dtype=float)
    paths = root_to_leaf_paths(xyz, edges)
    for path in paths:
        curve = bezier_curve(xyz[path])
        smooth[path, :3] += curve
        smooth[path, 3] += 1.0
    missing = smooth[:, 3] == 0
    smooth[~missing, :3] /= smooth[~missing, 3:4]
    smooth[missing, :3] = xyz[missing]
    smooth[:, 3] = skeleton_points[:, 3] if skeleton_points.shape[1] > 3 else 0.0
    return smooth, edges


def compute_skeleton_from_seeds(
    points: np.ndarray,
    seeds: np.ndarray,
    growth_vectors: np.ndarray,
    step_len: float = 0.5,
    steps: int = 20,
    grid_resolution: tuple[int, int, int] = (15, 15, 15),
    redundant_filter_num: int = 10,
    bezier_filter_num: int = 15,
    topology_neighbor_k: int = 40,
    topology_connector_scale: float = 18.0,
    topology_support_scale: float = 6.0,
    topology_upward_penalty: float = 4.0,
    progress: callable | None = None,
) -> tuple[np.ndarray, SkeletonResult]:
    labels = assign_labels(points, seeds)
    stream = stream_line_tracing(
        points,
        seeds,
        labels,
        growth_vectors,
        grid_resolution,
        step_len,
        steps,
        progress,
    )
    if stream.shape[0] < 2:
        empty = np.zeros((0, 2), dtype=np.int64)
        result = SkeletonResult(stream, empty, stream, empty, stream, [])
        return labels, result

    if progress:
        progress("生成拓扑约束骨架图")
    raw_edges = build_topology_aware_mst(
        stream,
        points,
        neighbor_k=topology_neighbor_k,
        connector_scale=topology_connector_scale,
        support_scale=topology_support_scale,
        upward_penalty=topology_upward_penalty,
    )

    if progress:
        progress("冗余路径过滤")
    refined_points, refined_edges = redundant_filter(stream, raw_edges, redundant_filter_num)

    if progress:
        progress("生长趋势过滤")
    refined_points, refined_edges = growth_trend_filter(refined_points, refined_edges)

    if progress:
        progress("Bezier 平滑")
    smooth_points, smooth_edges = bezier_smooth(refined_points, refined_edges, bezier_filter_num)
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    smooth_points[:, :3] = np.clip(smooth_points[:, :3], bounds_min, bounds_max)
    paths = root_to_leaf_paths(smooth_points[:, :3], smooth_edges)

    return labels, SkeletonResult(
        stream_points=stream,
        raw_edges=raw_edges,
        refined_points=smooth_points,
        refined_edges=smooth_edges,
        smooth_points=smooth_points,
        root_to_leaf=paths,
    )
