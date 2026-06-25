from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
(ROOT / "cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from pipeline.config import PipelineConfig  # noqa: E402
from pipeline.runner import SkeletonPipeline  # noqa: E402


def main() -> int:
    cfg = PipelineConfig()
    pipe = SkeletonPipeline(cfg)

    def progress(message: str) -> None:
        print(message, flush=True)

    state = pipe.run_all(progress)
    paths = pipe.export_outputs(ROOT / "outputs")
    point_count = 0 if state.points is None else state.points.shape[0]
    seed_count = 0 if state.seeds is None else state.seeds.shape[0]
    node_count = 0 if state.skeleton is None else state.skeleton.smooth_points.shape[0]
    edge_count = 0 if state.skeleton is None else state.skeleton.refined_edges.shape[0]

    print(f"points={point_count}")
    print(f"seeds={seed_count}")
    print(f"skeleton_nodes={node_count}")
    print(f"skeleton_edges={edge_count}")
    for key, path in paths.items():
        print(f"{key}: {path}")

    if point_count == 0 or seed_count == 0 or node_count == 0 or edge_count == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
