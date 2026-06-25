from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from io_utils.point_io import load_point_cloud, save_csv, save_dict_csv, save_json, save_obj_lines

from .config import PipelineConfig
from .geometry import compute_growth_vectors
from .metrics import BranchMetricsResult, compute_branch_metrics
from .preprocess import preprocess_points
from .seeds import assign_labels, set_seed_points_basic, set_seed_points_centered
from .skeleton import SkeletonResult, compute_skeleton_from_seeds


@dataclass(slots=True)
class PipelineState:
    raw_points: np.ndarray | None = None
    points: np.ndarray | None = None
    mean_position: np.ndarray | None = None
    z_offset: float | None = None
    growth_vectors: np.ndarray | None = None
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    seeds: np.ndarray | None = None
    labels: np.ndarray | None = None
    skeleton: SkeletonResult | None = None
    branch_metrics: BranchMetricsResult | None = None


class SkeletonPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.state = PipelineState()

    def load_and_preprocess(self, progress: callable | None = None) -> np.ndarray:
        if progress:
            progress(f"加载点云: {self.config.data_path}")
        raw = load_point_cloud(self.config.data_path)
        if progress:
            progress(f"下采样 gridStep={self.config.grid_step}")
        points, mean_pos, z_min = preprocess_points(raw, self.config.grid_step)
        self.state = PipelineState(
            raw_points=raw,
            points=points,
            mean_position=mean_pos,
            z_offset=z_min,
        )
        return points

    def compute_vectors(self, progress: callable | None = None) -> np.ndarray:
        points = self._require_points()
        vectors, aux = compute_growth_vectors(
            points,
            k=self.config.triangulation_k,
            heat_time=self.config.heat_time,
            progress=progress,
        )
        self.state.growth_vectors = vectors
        self.state.aux.update(aux)
        return vectors

    def generate_seeds(self, progress: callable | None = None) -> tuple[np.ndarray, np.ndarray]:
        points = self._require_points()
        vectors = self._require_vectors()
        if self.config.seed_mode == "centered":
            seeds, labels, info = set_seed_points_centered(
                points,
                vectors,
                self.config.init_seed_count,
                self.config.grid_step,
                random_seed=self.config.random_seed,
                progress=progress,
            )
        else:
            seeds, labels = set_seed_points_basic(
                points,
                self.config.init_seed_count,
                self.config.grid_step,
                progress=progress,
            )
            info = {"shifts": []}
        self.state.seeds = seeds
        self.state.labels = labels
        self.state.aux["seed_shifts"] = np.asarray(info.get("shifts", []), dtype=float)
        return seeds, labels

    def set_seeds(self, seeds: np.ndarray) -> np.ndarray:
        points = self._require_points()
        seeds = np.asarray(seeds, dtype=float).reshape(-1, 3)
        self.state.seeds = seeds
        self.state.labels = assign_labels(points, seeds)
        return self.state.labels

    def recompute_skeleton(self, progress: callable | None = None) -> SkeletonResult:
        points = self._require_points()
        vectors = self._require_vectors()
        seeds = self._require_seeds()
        labels, skeleton = compute_skeleton_from_seeds(
            points,
            seeds,
            vectors,
            step_len=self.config.streamline_step,
            steps=self.config.streamline_steps,
            grid_resolution=self.config.streamline_grid,
            redundant_filter_num=self.config.redundant_filter_num,
            bezier_filter_num=self.config.bezier_filter_num,
            progress=progress,
        )
        self.state.labels = labels
        self.state.skeleton = skeleton
        self.state.branch_metrics = None
        return skeleton

    def compute_branch_metrics(self) -> BranchMetricsResult:
        points = self._require_points()
        skeleton = self.state.skeleton
        if skeleton is None or skeleton.smooth_points.size == 0 or skeleton.refined_edges.size == 0:
            raise RuntimeError("No skeleton is available for branch metrics.")
        metrics = compute_branch_metrics(points, skeleton.smooth_points[:, :3], skeleton.refined_edges)
        self.state.branch_metrics = metrics
        return metrics

    def run_all(self, progress: callable | None = None) -> PipelineState:
        self.load_and_preprocess(progress)
        self.compute_vectors(progress)
        self.generate_seeds(progress)
        self.recompute_skeleton(progress)
        return self.state

    def export_outputs(self, output_dir: str | Path | None = None) -> dict[str, Path]:
        output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        seeds = self._require_seeds()
        skeleton = self.state.skeleton
        if skeleton is None or skeleton.smooth_points.size == 0:
            raise RuntimeError("No skeleton is available to export.")

        seed_path = output_dir / "tree_5_seeds_corrected.csv"
        node_path = output_dir / "tree_5_skeleton_nodes.csv"
        edge_path = output_dir / "tree_5_skeleton_edges.csv"
        obj_path = output_dir / "tree_5_skeleton.obj"
        summary_path = output_dir / "tree_5_branch_summary.json"
        branch_path = output_dir / "tree_5_branch_metrics.csv"

        save_csv(seed_path, seeds, "x,y,z")
        save_csv(node_path, skeleton.smooth_points[:, :3], "x,y,z")
        save_csv(edge_path, skeleton.refined_edges.astype(int), "source_index,target_index")
        save_obj_lines(obj_path, skeleton.smooth_points[:, :3], skeleton.refined_edges)
        if self.state.branch_metrics is None:
            self.compute_branch_metrics()
        if self.state.branch_metrics is not None:
            save_json(summary_path, self.state.branch_metrics.summary)
            save_dict_csv(branch_path, self.state.branch_metrics.branches)
        return {
            "seeds": seed_path,
            "nodes": node_path,
            "edges": edge_path,
            "obj": obj_path,
            "branch_summary": summary_path,
            "branch_metrics": branch_path,
        }

    def _require_points(self) -> np.ndarray:
        if self.state.points is None:
            raise RuntimeError("Point cloud has not been loaded.")
        return self.state.points

    def _require_vectors(self) -> np.ndarray:
        if self.state.growth_vectors is None:
            raise RuntimeError("Growth vectors have not been computed.")
        return self.state.growth_vectors

    def _require_seeds(self) -> np.ndarray:
        if self.state.seeds is None:
            raise RuntimeError("Seed points have not been generated.")
        return self.state.seeds
