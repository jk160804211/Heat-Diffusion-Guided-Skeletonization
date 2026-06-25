from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class BranchMetricsResult:
    summary: dict[str, float | int]
    branches: list[dict[str, float | int | str]]


def _adjacency(n: int, edges: np.ndarray) -> list[list[int]]:
    adj = [[] for _ in range(n)]
    for u, v in np.asarray(edges, dtype=int):
        if 0 <= u < n and 0 <= v < n and u != v:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def _root_tree(nodes: np.ndarray, edges: np.ndarray) -> tuple[int, np.ndarray, list[list[int]], np.ndarray]:
    n = nodes.shape[0]
    root = int(np.argmin(nodes[:, 2]))
    adj = _adjacency(n, edges)
    parent = np.full(n, -1, dtype=int)
    children = [[] for _ in range(n)]
    depth = np.zeros(n, dtype=int)
    q: deque[int] = deque([root])
    seen = np.zeros(n, dtype=bool)
    seen[root] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            parent[v] = u
            children[u].append(v)
            depth[v] = depth[u] + 1
            q.append(v)
    return root, parent, children, depth


def _node_strahler(children: list[list[int]], root: int) -> np.ndarray:
    order = np.zeros(len(children), dtype=int)
    stack: list[tuple[int, bool]] = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            if not children[node]:
                order[node] = 1
            else:
                child_orders = [int(order[c]) for c in children[node]]
                m = max(child_orders)
                order[node] = m + 1 if child_orders.count(m) >= 2 else m
            continue
        stack.append((node, True))
        for child in children[node]:
            stack.append((child, False))
    return order


def _path_length(nodes: np.ndarray, path: list[int]) -> float:
    if len(path) < 2:
        return 0.0
    p = nodes[path]
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    c = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _main_stem_center_at_z(
    nodes: np.ndarray,
    children: list[list[int]],
    strahler: np.ndarray,
    root: int,
    target_z: float,
) -> np.ndarray | None:
    if nodes.size == 0 or not children:
        return None
    path = [int(root)]
    cur = int(root)
    seen: set[int] = {cur}
    while children[cur]:
        nxt = max(
            children[cur],
            key=lambda c: (int(strahler[c]) if c < strahler.shape[0] else 0, float(nodes[c, 2]), -len(children[c])),
        )
        nxt = int(nxt)
        if nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        cur = nxt

    if len(path) < 2:
        return nodes[int(root)].copy()

    best = nodes[path[int(np.argmin(np.abs(nodes[path, 2] - target_z)))]].copy()
    for a, b in zip(path[:-1], path[1:]):
        p0 = nodes[a]
        p1 = nodes[b]
        z0 = float(p0[2])
        z1 = float(p1[2])
        if min(z0, z1) <= target_z <= max(z0, z1) and abs(z1 - z0) > 1e-9:
            t = (target_z - z0) / (z1 - z0)
            return p0 + t * (p1 - p0)
    return best


def _estimate_dbh(
    points: np.ndarray,
    nodes: np.ndarray,
    children: list[list[int]],
    strahler: np.ndarray,
    root: int,
) -> dict[str, float | int]:
    z0 = float(points[:, 2].min())
    height = float(points[:, 2].max() - z0)
    dbh_height = 1.3 if height > 2.6 else 0.15 * height
    target_z = z0 + dbh_height
    band = max(0.18, 0.025 * height)
    mask = np.abs(points[:, 2] - target_z) <= band
    if np.count_nonzero(mask) < 20:
        mask = np.abs(points[:, 2] - target_z) <= max(2 * band, 0.055 * height)
    sample = points[mask] if np.count_nonzero(mask) >= 6 else points
    stem_center = _main_stem_center_at_z(nodes, children, strahler, root, target_z)
    center = np.median(sample[:, :2], axis=0)
    if stem_center is not None and np.all(np.isfinite(stem_center)):
        radial_to_stem = np.linalg.norm(sample[:, :2] - stem_center[:2], axis=1)
        if np.count_nonzero(np.isfinite(radial_to_stem)) >= 6:
            center = stem_center[:2]
    radius = np.linalg.norm(sample[:, :2] - center, axis=1)
    radius = radius[np.isfinite(radius)]
    dbh_radius = float(np.percentile(radius, 62)) if radius.size else max(0.032 * height, 1e-6)
    return {
        "dbh_height": float(dbh_height),
        "dbh_z": float(target_z),
        "dbh_center_x": float(center[0]),
        "dbh_center_y": float(center[1]),
        "dbh_radius": float(dbh_radius),
        "dbh": float(2.0 * dbh_radius),
        "dbh_sample_count": int(sample.shape[0]),
    }


def _radius_near_point(
    point: np.ndarray,
    direction: np.ndarray,
    cloud_tree: cKDTree,
    points: np.ndarray,
    fallback: float,
    search_radius: float,
) -> float:
    idx = cloud_tree.query_ball_point(point, r=search_radius)
    if len(idx) < 8:
        return fallback
    sample = points[np.asarray(idx, dtype=int)]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    axial = (sample - point) @ direction
    proj = point + np.outer(axial, direction)
    radial = np.linalg.norm(sample - proj, axis=1)
    radial = radial[np.isfinite(radial) & (radial > 0)]
    if radial.size < 6:
        return fallback
    return float(np.percentile(radial, 58))


def _build_branches(
    nodes: np.ndarray,
    edges: np.ndarray,
    parent: np.ndarray,
    children: list[list[int]],
    strahler: np.ndarray,
) -> list[dict[str, object]]:
    root = int(np.argmin(nodes[:, 2]))
    branches: list[dict[str, object]] = []
    branch_by_end: dict[int, int] = {}
    queue: deque[tuple[int, int | None, int]] = deque((child, None, 0) for child in children[root])
    while queue:
        start_child, parent_branch, topo_order = queue.popleft()
        start = int(parent[start_child])
        path = [start, int(start_child)]
        cur = int(start_child)
        while len(children[cur]) == 1:
            cur = int(children[cur][0])
            path.append(cur)
        branch_id = len(branches)
        branch_by_end[cur] = branch_id
        branches.append(
            {
                "branch_id": branch_id,
                "parent_branch_id": -1 if parent_branch is None else parent_branch,
                "topological_order": topo_order,
                "strahler_order": int(strahler[cur]),
                "start_node": int(path[0]),
                "end_node": int(path[-1]),
                "path": path,
            }
        )
        for child in children[cur]:
            queue.append((int(child), branch_id, topo_order + 1))
    return branches


def compute_branch_metrics(points: np.ndarray, nodes: np.ndarray, edges: np.ndarray) -> BranchMetricsResult:
    points = np.asarray(points, dtype=float)
    nodes = np.asarray(nodes, dtype=float)
    edges = np.asarray(edges, dtype=int)
    if points.size == 0 or nodes.size == 0 or edges.size == 0:
        return BranchMetricsResult({}, [])

    root, parent, children, depth = _root_tree(nodes, edges)
    strahler = _node_strahler(children, root)
    branches = _build_branches(nodes, edges, parent, children, strahler)

    dbh_info = _estimate_dbh(points, nodes, children, strahler, root)
    dbh_height = float(dbh_info["dbh_height"])
    dbh_radius = float(dbh_info["dbh_radius"])
    dbh = float(dbh_info["dbh"])
    z_min = float(points[:, 2].min())
    height = float(points[:, 2].max() - z_min)
    basal_area = math.pi * dbh_radius**2
    cloud_tree = cKDTree(points)
    search_radius = max(0.35, 0.055 * max(height, 1.0))
    tip_radius = max(0.006 * max(height, 1.0), 0.11 * dbh_radius)
    path_dist = np.zeros(nodes.shape[0], dtype=float)
    order_nodes = np.argsort(depth)
    for node in order_nodes:
        p = parent[node]
        if p >= 0:
            path_dist[node] = path_dist[p] + np.linalg.norm(nodes[node] - nodes[p])
    max_dist = max(float(np.max(path_dist)), 1e-6)

    branch_rows: list[dict[str, float | int | str]] = []
    for raw in branches:
        path = list(raw["path"])  # type: ignore[arg-type]
        p = nodes[path]
        length = _path_length(nodes, path)
        chord = float(np.linalg.norm(p[-1] - p[0]))
        direction = p[-1] - p[0]
        horizontal = float(np.linalg.norm(direction[:2]))
        vertical = float(direction[2])
        inclination = _angle_deg(direction, np.array([0.0, 0.0, 1.0]))
        elevation = float(np.degrees(np.arctan2(vertical, max(horizontal, 1e-12))))
        azimuth = float((np.degrees(np.arctan2(direction[1], direction[0])) + 360.0) % 360.0)
        parent_id = int(raw["parent_branch_id"])
        parent_angle = float("nan")
        if parent_id >= 0 and parent_id < len(branch_rows):
            parent_vec = nodes[int(branch_rows[parent_id]["end_node"])] - nodes[int(branch_rows[parent_id]["start_node"])]
            parent_angle = _angle_deg(direction, parent_vec)

        rel0 = min(max(path_dist[path[0]] / max_dist, 0.0), 1.0)
        rel1 = min(max(path_dist[path[-1]] / max_dist, 0.0), 1.0)
        order_factor = 0.65 ** int(raw["topological_order"])
        taper0 = max((1.0 - rel0) ** 0.65, 0.0)
        taper1 = max((1.0 - rel1) ** 0.65, 0.0)
        fallback0 = max(dbh_radius * taper0 * order_factor, tip_radius)
        fallback1 = max(dbh_radius * taper1 * order_factor, tip_radius)
        radius_start = _radius_near_point(p[0], direction, cloud_tree, points, fallback0, search_radius)
        radius_end = min(radius_start, _radius_near_point(p[-1], direction, cloud_tree, points, fallback1, search_radius))
        radius_start = max(radius_start, radius_end, tip_radius)
        radius_end = max(radius_end, tip_radius)
        taper_rate = (radius_start - radius_end) / max(length, 1e-12)
        volume = math.pi * length * (radius_start**2 + radius_start * radius_end + radius_end**2) / 3.0
        slant = math.sqrt((radius_start - radius_end) ** 2 + length**2)
        surface_area = math.pi * (radius_start + radius_end) * slant

        branch_rows.append(
            {
                "branch_id": int(raw["branch_id"]),
                "parent_branch_id": parent_id,
                "topological_order": int(raw["topological_order"]),
                "strahler_order": int(raw["strahler_order"]),
                "start_node": int(raw["start_node"]),
                "end_node": int(raw["end_node"]),
                "path_nodes": ";".join(str(int(i)) for i in path),
                "base_height": float(p[0, 2]),
                "tip_height": float(p[-1, 2]),
                "length": length,
                "chord_length": chord,
                "tortuosity": float(length / max(chord, 1e-12)),
                "horizontal_projection": horizontal,
                "vertical_rise": vertical,
                "inclination_from_vertical_deg": inclination,
                "elevation_deg": elevation,
                "azimuth_deg": azimuth,
                "parent_angle_deg": parent_angle,
                "radius_start": radius_start,
                "radius_end": radius_end,
                "diameter_start": 2.0 * radius_start,
                "diameter_end": 2.0 * radius_end,
                "taper_rate": taper_rate,
                "volume": volume,
                "surface_area": surface_area,
                "slenderness": float(length / max(2.0 * radius_start, 1e-12)),
            }
        )

    tips = int(sum(1 for c in children if len(c) == 0))
    bifurcations = int(sum(1 for c in children if len(c) >= 2))
    branch_base_heights = [r["base_height"] for r in branch_rows if r["parent_branch_id"] >= 0]
    crown_base = float(min(branch_base_heights)) if branch_base_heights else z_min
    crown_len = max(float(points[:, 2].max()) - crown_base, 0.0)
    spread_x = float(points[:, 0].max() - points[:, 0].min())
    spread_y = float(points[:, 1].max() - points[:, 1].min())
    crown_area = math.pi * (spread_x / 2.0) * (spread_y / 2.0)
    crown_volume_ellipsoid = 4.0 / 3.0 * math.pi * (spread_x / 2.0) * (spread_y / 2.0) * (crown_len / 2.0)
    total_length = float(sum(r["length"] for r in branch_rows))
    total_volume = float(sum(r["volume"] for r in branch_rows))
    total_surface = float(sum(r["surface_area"] for r in branch_rows))
    angles = np.array([r["parent_angle_deg"] for r in branch_rows], dtype=float)
    angles = angles[np.isfinite(angles)]

    summary: dict[str, float | int] = {
        "tree_height": height,
        "dbh_height": dbh_height,
        "dbh_z": float(dbh_info["dbh_z"]),
        "dbh_center_x": float(dbh_info["dbh_center_x"]),
        "dbh_center_y": float(dbh_info["dbh_center_y"]),
        "dbh": dbh,
        "dbh_radius": dbh_radius,
        "dbh_sample_count": int(dbh_info["dbh_sample_count"]),
        "basal_area": basal_area,
        "crown_base_height": crown_base,
        "crown_length": crown_len,
        "crown_ratio": crown_len / max(height, 1e-12),
        "crown_spread_x": spread_x,
        "crown_spread_y": spread_y,
        "crown_projected_area_ellipse": crown_area,
        "crown_volume_ellipsoid": crown_volume_ellipsoid,
        "branch_count": len(branch_rows),
        "terminal_tip_count": tips,
        "bifurcation_count": bifurcations,
        "max_topological_order": max((int(r["topological_order"]) for r in branch_rows), default=0),
        "max_strahler_order": int(np.max(strahler)) if strahler.size else 0,
        "total_branch_length": total_length,
        "total_wood_volume_frustum": total_volume,
        "total_surface_area_frustum": total_surface,
        "mean_parent_branch_angle_deg": float(np.mean(angles)) if angles.size else float("nan"),
        "max_parent_branch_angle_deg": float(np.max(angles)) if angles.size else float("nan"),
    }
    return BranchMetricsResult(summary, branch_rows)
