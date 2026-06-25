from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "cache"))
(ROOT / "cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    QTimer.singleShot(100, win.run_preprocess)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
