from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_PATH = Path(
    "/Users/jiangkang/Desktop/code/LiDAR data/IEEE TRANS data/tree_5.txt"
)


@dataclass(slots=True)
class PipelineConfig:
    data_path: Path = DEFAULT_DATA_PATH
    grid_step: float = 0.1
    init_seed_count: int = 450
    seed_mode: str = "main3"
    triangulation_k: int = 10
    heat_time: float = 1000.0
    streamline_grid: tuple[int, int, int] = (15, 15, 15)
    streamline_step: float = 0.5
    streamline_steps: int = 20
    redundant_filter_num: int = 10
    bezier_filter_num: int = 15
    random_seed: int = 18
    max_points_for_vectors: int = 4500
