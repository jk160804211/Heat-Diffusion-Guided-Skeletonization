from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QGroupBox,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import cKDTree

from pipeline.config import PipelineConfig
from pipeline.runner import SkeletonPipeline


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Skeleton Extraction GUI - Seed Correction")
        self.resize(1480, 920)

        self.pipeline = SkeletonPipeline(PipelineConfig())
        self.selected_seed: int | None = None
        self.seed_scatter = None
        self._updating_table = False
        self._updating_spins = False
        self._history: list[np.ndarray] = []
        self._future: list[np.ndarray] = []
        self._dragging = False
        self._drag_last: tuple[float, float] | None = None
        self._panning = False
        self._pan_last: tuple[float, float] | None = None
        self._rotating = False
        self._rotate_last: tuple[float, float] | None = None
        self._manual_axes_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
        self.metric_summary: dict = {}
        self.metric_branches: list[dict] = []
        self.branch_metric_columns: list[str] = []
        self.metric_overlay: tuple[str, str, int | None] | None = None

        self._build_ui()
        self._connect_canvas_events()
        self.statusBar().showMessage("Ready")

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([320, 850, 360])
        self._prepare_splitter(splitter)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        export_action = QAction("导出结果", self)
        export_action.triggered.connect(self.export_outputs)
        self.menuBar().addAction(export_action)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.path_edit = QLineEdit(str(self.pipeline.config.data_path))
        self.path_edit.setMinimumWidth(260)
        browse_btn = QPushButton("选择数据")
        browse_btn.clicked.connect(self.choose_data_file)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addWidget(QLabel("数据路径"))
        layout.addLayout(path_row)

        form = QFormLayout()
        self.grid_step_spin = QDoubleSpinBox()
        self.grid_step_spin.setRange(0.001, 10.0)
        self.grid_step_spin.setDecimals(4)
        self.grid_step_spin.setValue(self.pipeline.config.grid_step)
        self.grid_step_spin.setSingleStep(0.05)

        self.seed_count_spin = QSpinBox()
        self.seed_count_spin.setRange(1, 5000)
        self.seed_count_spin.setValue(self.pipeline.config.init_seed_count)

        self.step_len_spin = QDoubleSpinBox()
        self.step_len_spin.setRange(0.001, 10.0)
        self.step_len_spin.setDecimals(3)
        self.step_len_spin.setValue(self.pipeline.config.streamline_step)
        self.step_len_spin.setSingleStep(0.05)

        self.trace_steps_spin = QSpinBox()
        self.trace_steps_spin.setRange(1, 500)
        self.trace_steps_spin.setValue(self.pipeline.config.streamline_steps)

        self.seed_mode_combo = QComboBox()
        self.seed_mode_combo.addItem("main3: setSeedPoints", "main3")
        self.seed_mode_combo.addItem("centered: setSeedPoints_centered", "centered")

        form.addRow("gridStep", self.grid_step_spin)
        form.addRow("初始种子数", self.seed_count_spin)
        form.addRow("种子算法", self.seed_mode_combo)
        form.addRow("流线步长", self.step_len_spin)
        form.addRow("流线步数", self.trace_steps_spin)
        layout.addLayout(form)

        self.load_btn = QPushButton("1. 加载并预处理")
        self.vector_btn = QPushButton("2. 计算热扩散/生长向量")
        self.seed_btn = QPushButton("3. 自动生成种子")
        self.skeleton_btn = QPushButton("4. 用当前种子重算骨架")
        self.export_btn = QPushButton("导出 outputs")

        self.load_btn.clicked.connect(self.run_preprocess)
        self.vector_btn.clicked.connect(self.run_vectors)
        self.seed_btn.clicked.connect(self.generate_seeds)
        self.skeleton_btn.clicked.connect(self.recompute_skeleton)
        self.export_btn.clicked.connect(self.export_outputs)

        for btn in (
            self.load_btn,
            self.vector_btn,
            self.seed_btn,
            self.skeleton_btn,
            self.export_btn,
        ):
            layout.addWidget(btn)

        layout.addSpacing(12)
        self.stage_combo = QComboBox()
        self.stage_combo.addItems(
            [
                "1 点云预处理",
                "2 生长向量",
                "3 标签和种子",
                "4 流线和向量",
                "5 初始流线图",
                "6 初始骨架",
                "7 精修骨架",
            ]
        )
        self.stage_combo.currentIndexChanged.connect(self.update_plot)
        layout.addWidget(QLabel("可视化阶段"))
        layout.addWidget(self.stage_combo)

        style_form = QFormLayout()
        self.point_visible_check = QCheckBox("显示")
        self.point_visible_check.setChecked(True)
        self.point_visible_check.stateChanged.connect(self.update_plot)

        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(0.1, 30.0)
        self.point_size_spin.setDecimals(1)
        self.point_size_spin.setSingleStep(0.5)
        self.point_size_spin.setValue(5.0)
        self.point_size_spin.valueChanged.connect(self.update_plot)

        self.point_alpha_spin = QDoubleSpinBox()
        self.point_alpha_spin.setRange(0.02, 1.0)
        self.point_alpha_spin.setDecimals(2)
        self.point_alpha_spin.setSingleStep(0.05)
        self.point_alpha_spin.setValue(0.54)
        self.point_alpha_spin.valueChanged.connect(self.update_plot)

        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(500, 200000)
        self.max_points_spin.setSingleStep(500)
        self.max_points_spin.setValue(12000)
        self.max_points_spin.valueChanged.connect(self.update_plot)

        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(
            [
                "阶段自动",
                "灰色",
                "标签颜色(计算)",
                "热扩散颜色(计算)",
                "生长强度颜色(计算)",
                "Z高度自动调节",
            ]
        )
        self.color_mode_combo.currentIndexChanged.connect(self.update_plot)
        style_form.addRow("点云显示", self.point_visible_check)
        style_form.addRow("点云大小", self.point_size_spin)
        style_form.addRow("点云透明度", self.point_alpha_spin)
        style_form.addRow("最多显示点", self.max_points_spin)
        style_form.addRow("点云颜色", self.color_mode_combo)
        layout.addLayout(style_form)

        self.show_labels_check = QCheckBox("显示种子标签色")
        self.show_labels_check.stateChanged.connect(self.update_plot)
        self.show_stream_check = QCheckBox("显示候选流线")
        self.show_stream_check.stateChanged.connect(self.update_plot)
        layout.addWidget(self.show_labels_check)
        layout.addWidget(self.show_stream_check)

        layout.addStretch(1)
        return panel

    def _build_canvas_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.figure = Figure(figsize=(8, 7), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, panel)
        self.ax = self.figure.add_subplot(111, projection="3d")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_seed_panel())
        splitter.addWidget(self._build_metrics_panel())
        splitter.setSizes([520, 420])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self._prepare_splitter(splitter)
        splitter.setChildrenCollapsible(False)
        return splitter

    def _prepare_splitter(self, splitter: QSplitter) -> None:
        splitter.setHandleWidth(12)
        splitter.setOpaqueResize(True)
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background: #aeb4bf;
                border: 1px solid #7f8794;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background: #6f7b8f;
            }
            """
        )

    def _build_seed_panel(self) -> QWidget:
        panel = QGroupBox("种子点编辑")
        layout = QVBoxLayout(panel)
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        seed_style = QFormLayout()
        self.seed_visible_check = QCheckBox("显示")
        self.seed_visible_check.setChecked(True)
        self.seed_visible_check.stateChanged.connect(self.update_plot)

        self.seed_size_spin = QDoubleSpinBox()
        self.seed_size_spin.setRange(5.0, 300.0)
        self.seed_size_spin.setDecimals(1)
        self.seed_size_spin.setSingleStep(5.0)
        self.seed_size_spin.setValue(42.0)
        self.seed_size_spin.valueChanged.connect(self.update_plot)

        self.seed_color_combo = QComboBox()
        self.seed_color_combo.addItems(["红色", "蓝色", "绿色", "黑色", "白色", "标签颜色"])
        self.seed_color_combo.currentIndexChanged.connect(self.update_plot)
        seed_style.addRow("种子显示", self.seed_visible_check)
        seed_style.addRow("种子大小", self.seed_size_spin)
        seed_style.addRow("种子颜色", self.seed_color_combo)
        controls_layout.addLayout(seed_style)

        self.seed_table = QTableWidget(0, 5)
        self.seed_table.setHorizontalHeaderLabels(["ID", "X", "Y", "Z", "删除"])
        self.seed_table.setMinimumHeight(70)
        self.seed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.seed_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.seed_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.seed_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.seed_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.seed_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.seed_table.horizontalHeader().setStretchLastSection(True)
        self.seed_table.itemSelectionChanged.connect(self._table_selection_changed)
        self.seed_table.itemChanged.connect(self._seed_table_item_changed)
        table_box = QWidget()
        table_box.setMinimumHeight(80)
        table_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(QLabel("种子点"))
        table_layout.addWidget(self.seed_table, 1)

        button_row = QHBoxLayout()
        self.add_seed_btn = QPushButton("添加")
        self.delete_seed_btn = QPushButton("删除选中")
        self.snap_seed_btn = QPushButton("吸附最近点")
        button_row.addWidget(self.add_seed_btn)
        button_row.addWidget(self.delete_seed_btn)
        button_row.addWidget(self.snap_seed_btn)
        controls_layout.addLayout(button_row)

        undo_row = QHBoxLayout()
        self.undo_btn = QPushButton("撤销")
        self.redo_btn = QPushButton("重做")
        undo_row.addWidget(self.undo_btn)
        undo_row.addWidget(self.redo_btn)
        controls_layout.addLayout(undo_row)

        self.add_seed_btn.clicked.connect(self.add_seed)
        self.delete_seed_btn.clicked.connect(self.delete_selected_seed)
        self.snap_seed_btn.clicked.connect(self.snap_selected_seed)
        self.undo_btn.clicked.connect(self.undo_seed_edit)
        self.redo_btn.clicked.connect(self.redo_seed_edit)

        controls_layout.addWidget(QLabel("坐标微调"))
        form = QFormLayout()
        self.x_spin = self._coord_spin()
        self.y_spin = self._coord_spin()
        self.z_spin = self._coord_spin()
        self.x_spin.valueChanged.connect(lambda value: self._coord_spin_changed(0, value))
        self.y_spin.valueChanged.connect(lambda value: self._coord_spin_changed(1, value))
        self.z_spin.valueChanged.connect(lambda value: self._coord_spin_changed(2, value))
        form.addRow("X", self.x_spin)
        form.addRow("Y", self.y_spin)
        form.addRow("Z", self.z_spin)
        controls_layout.addLayout(form)

        drag_row = QHBoxLayout()
        self.drag_check = QCheckBox("启用拖动")
        self.drag_plane = QComboBox()
        self.drag_plane.addItems(["XY", "XZ", "YZ"])
        self.drag_mode = QComboBox()
        self.drag_mode.addItems(["平面自由", "最近点吸附", "局部中心吸附"])
        drag_row.addWidget(self.drag_check)
        drag_row.addWidget(QLabel("投影面"))
        drag_row.addWidget(self.drag_plane)
        drag_row.addWidget(QLabel("模式"))
        drag_row.addWidget(self.drag_mode)
        controls_layout.addLayout(drag_row)

        hint = QLabel("提示：点击红色种子或表格行选中；拖动支持平面自由、最近点吸附、局部中心吸附三种模式。")
        hint.setWordWrap(True)
        controls_layout.addWidget(hint)
        controls_layout.addStretch(1)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumHeight(110)
        controls_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        seed_splitter = QSplitter(Qt.Orientation.Vertical)
        seed_splitter.addWidget(controls_scroll)
        seed_splitter.addWidget(table_box)
        seed_splitter.setSizes([260, 380])
        seed_splitter.setStretchFactor(0, 0)
        seed_splitter.setStretchFactor(1, 1)
        self._prepare_splitter(seed_splitter)
        seed_splitter.setChildrenCollapsible(False)
        layout.addWidget(seed_splitter, 1)

        return panel

    def _build_metrics_panel(self) -> QWidget:
        panel = QGroupBox("枝干参数反演")
        layout = QVBoxLayout(panel)

        self.metrics_btn = QPushButton("计算枝干参数")
        self.metrics_btn.clicked.connect(self.compute_branch_metrics)
        self.metric_overlay_check = QCheckBox("显示参数可视化")
        self.metric_overlay_check.setChecked(True)
        self.metric_overlay_check.stateChanged.connect(self.update_plot)
        self.export_metrics_btn = QPushButton("导出反演参数")
        self.export_metrics_btn.clicked.connect(self.export_metrics)
        metric_buttons = QHBoxLayout()
        metric_buttons.addWidget(self.metrics_btn)
        metric_buttons.addWidget(self.export_metrics_btn)
        layout.addLayout(metric_buttons)
        layout.addWidget(self.metric_overlay_check)

        table_splitter = QSplitter(Qt.Orientation.Vertical)
        summary_box = QWidget()
        summary_box.setMinimumHeight(70)
        summary_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.addWidget(QLabel("树木/冠层汇总参数"))
        self.summary_table = QTableWidget(0, 2)
        self.summary_table.setHorizontalHeaderLabels(["参数", "值"])
        self.summary_table.setMinimumHeight(45)
        self.summary_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        self.summary_table.itemSelectionChanged.connect(self._summary_metric_selected)
        summary_layout.addWidget(self.summary_table)

        branch_box = QWidget()
        branch_box.setMinimumHeight(70)
        branch_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        branch_layout = QVBoxLayout(branch_box)
        branch_layout.setContentsMargins(0, 0, 0, 0)
        branch_layout.addWidget(QLabel("拓扑枝段参数"))
        self.branch_table = QTableWidget(0, 8)
        self.branch_table.setMinimumHeight(45)
        self.branch_table.setHorizontalHeaderLabels(
            ["ID", "Parent", "Order", "Strahler", "Length", "Angle", "D_start", "Volume"]
        )
        self.branch_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.branch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.branch_table.horizontalHeader().setStretchLastSection(True)
        self.branch_table.itemSelectionChanged.connect(self._branch_metric_selected)
        branch_layout.addWidget(self.branch_table)
        table_splitter.addWidget(summary_box)
        table_splitter.addWidget(branch_box)
        table_splitter.setSizes([250, 430])
        table_splitter.setStretchFactor(0, 1)
        table_splitter.setStretchFactor(1, 1)
        self._prepare_splitter(table_splitter)
        table_splitter.setChildrenCollapsible(False)
        layout.addWidget(table_splitter, 1)
        return panel

    def _coord_spin(self) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setDecimals(6)
        spin.setSingleStep(0.01)
        return spin

    def _connect_canvas_events(self) -> None:
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

    def choose_data_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择点云文件",
            str(Path(self.path_edit.text()).parent),
            "Point cloud (*.txt *.csv *.mat);;All files (*)",
        )
        if path:
            self.path_edit.setText(path)

    def _sync_config_from_ui(self) -> None:
        cfg = self.pipeline.config
        cfg.data_path = Path(self.path_edit.text()).expanduser()
        cfg.grid_step = float(self.grid_step_spin.value())
        cfg.init_seed_count = int(self.seed_count_spin.value())
        cfg.seed_mode = str(self.seed_mode_combo.currentData())
        cfg.streamline_step = float(self.step_len_spin.value())
        cfg.streamline_steps = int(self.trace_steps_spin.value())

    def _run_busy(self, label: str, func) -> None:
        self._sync_config_from_ui()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._set_buttons_enabled(False)
        try:
            self.statusBar().showMessage(label)
            QApplication.processEvents()
            func()
        except Exception as exc:
            QMessageBox.critical(self, "流程错误", str(exc))
            self.statusBar().showMessage(f"错误: {exc}")
        finally:
            self._set_buttons_enabled(True)
            QApplication.restoreOverrideCursor()
            self._refresh_counts()

    def _set_buttons_enabled(self, enabled: bool) -> None:
        for btn in (
            self.load_btn,
            self.vector_btn,
            self.seed_btn,
            self.skeleton_btn,
            self.export_btn,
            self.add_seed_btn,
            self.delete_seed_btn,
            self.snap_seed_btn,
            self.undo_btn,
            self.redo_btn,
            self.metrics_btn,
            self.export_metrics_btn,
        ):
            btn.setEnabled(enabled)

    def _progress(self, message: str) -> None:
        self.statusBar().showMessage(message)
        QApplication.processEvents()

    def run_preprocess(self) -> None:
        def work() -> None:
            self.pipeline.load_and_preprocess(self._progress)
            self.selected_seed = None
            self._history.clear()
            self._future.clear()
            self._manual_axes_limits = None
            self.stage_combo.setCurrentIndex(0)
            self.update_seed_table()
            self.update_plot()

        self._run_busy("加载并预处理", work)

    def run_vectors(self) -> None:
        def work() -> None:
            if self.pipeline.state.points is None:
                self.pipeline.load_and_preprocess(self._progress)
            self.pipeline.compute_vectors(self._progress)
            self.stage_combo.setCurrentIndex(1)
            self.update_plot()

        self._run_busy("计算生长向量", work)

    def generate_seeds(self) -> None:
        def work() -> None:
            if self.pipeline.state.points is None:
                self.pipeline.load_and_preprocess(self._progress)
            if self.pipeline.state.growth_vectors is None:
                self.pipeline.compute_vectors(self._progress)
            self.pipeline.generate_seeds(self._progress)
            self.selected_seed = 0
            self._history.clear()
            self._future.clear()
            self.stage_combo.setCurrentIndex(2)
            self.update_seed_table()
            self.update_coord_spins()
            self.update_plot()

        self._run_busy("自动生成种子", work)

    def recompute_skeleton(self) -> None:
        def work() -> None:
            if self.pipeline.state.seeds is None:
                raise RuntimeError("请先生成或添加种子点。")
            self.pipeline.recompute_skeleton(self._progress)
            self.stage_combo.setCurrentIndex(6)
            self.update_plot()

        self._run_busy("重算骨架", work)

    def export_outputs(self) -> None:
        def work() -> None:
            paths = self.pipeline.export_outputs()
            joined = "\n".join(f"{k}: {v}" for k, v in paths.items())
            QMessageBox.information(self, "导出完成", joined)

        self._run_busy("导出结果", work)

    def compute_branch_metrics(self) -> None:
        def work() -> None:
            metrics = self.pipeline.compute_branch_metrics()
            self.update_metrics_tables(metrics.summary, metrics.branches)

        self._run_busy("计算枝干参数", work)

    def export_metrics(self) -> None:
        def work() -> None:
            if self.pipeline.state.branch_metrics is None:
                self.pipeline.compute_branch_metrics()
            paths = self.pipeline.export_outputs()
            QMessageBox.information(
                self,
                "反演参数导出完成",
                f"{paths.get('branch_summary')}\n{paths.get('branch_metrics')}",
            )

        self._run_busy("导出反演参数", work)

    def update_metrics_tables(self, summary: dict, branches: list[dict]) -> None:
        self.metric_summary = summary
        self.metric_branches = branches
        preferred = [
            "tree_height",
            "dbh_height",
            "dbh_z",
            "dbh_center_x",
            "dbh_center_y",
            "dbh",
            "dbh_radius",
            "dbh_sample_count",
            "basal_area",
            "crown_base_height",
            "crown_length",
            "crown_ratio",
            "crown_projected_area_ellipse",
            "crown_volume_ellipsoid",
            "branch_count",
            "terminal_tip_count",
            "bifurcation_count",
            "max_topological_order",
            "max_strahler_order",
            "total_branch_length",
            "total_wood_volume_frustum",
            "total_surface_area_frustum",
            "mean_parent_branch_angle_deg",
        ]
        ordered_summary = [k for k in preferred if k in summary]
        ordered_summary.extend(k for k in summary.keys() if k not in ordered_summary)
        rows = [(k, summary[k]) for k in ordered_summary]
        self.summary_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.summary_table.setItem(row, 0, QTableWidgetItem(str(key)))
            self.summary_table.setItem(row, 1, QTableWidgetItem(self._fmt_metric(value)))

        show = branches[:300]
        columns = self._branch_metric_columns(branches)
        self.branch_metric_columns = columns
        self.branch_table.setColumnCount(len(columns))
        self.branch_table.setHorizontalHeaderLabels(columns)
        self.branch_table.setRowCount(len(show))
        for row, branch in enumerate(show):
            for col, key in enumerate(columns):
                self.branch_table.setItem(row, col, QTableWidgetItem(self._fmt_metric(branch.get(key, ""))))
        self.branch_table.resizeColumnsToContents()

    def _branch_metric_columns(self, branches: list[dict]) -> list[str]:
        preferred = [
            "branch_id",
            "parent_branch_id",
            "topological_order",
            "strahler_order",
            "start_node",
            "end_node",
            "base_height",
            "tip_height",
            "length",
            "chord_length",
            "tortuosity",
            "horizontal_projection",
            "vertical_rise",
            "inclination_from_vertical_deg",
            "elevation_deg",
            "azimuth_deg",
            "parent_angle_deg",
            "radius_start",
            "radius_end",
            "diameter_start",
            "diameter_end",
            "taper_rate",
            "volume",
            "surface_area",
            "slenderness",
        ]
        available: set[str] = set()
        for branch in branches:
            available.update(str(k) for k in branch.keys())
        columns = [k for k in preferred if k in available]
        columns.extend(k for k in available if k not in columns and k != "path_nodes")
        return columns

    def _summary_metric_selected(self) -> None:
        rows = self.summary_table.selectionModel().selectedRows()
        if rows:
            row = rows[0].row()
        else:
            items = self.summary_table.selectedItems()
            if not items:
                return
            row = items[0].row()
        item = self.summary_table.item(row, 0)
        if item is None:
            return
        self.metric_overlay = ("summary", item.text(), None)
        self.metric_overlay_check.setChecked(True)
        self.update_plot()

    def _branch_metric_selected(self) -> None:
        items = self.branch_table.selectedItems()
        if not items:
            return
        row = items[0].row()
        if row < 0 or row >= len(self.metric_branches):
            return
        if not self.branch_metric_columns:
            return
        col = max(0, min(items[0].column(), len(self.branch_metric_columns) - 1))
        self.metric_overlay = ("branch", self.branch_metric_columns[col], row)
        self.metric_overlay_check.setChecked(True)
        self.update_plot()

    def _fmt_metric(self, value) -> str:
        if isinstance(value, float):
            if not np.isfinite(value):
                return ""
            return f"{value:.4f}"
        return str(value)

    def _push_seed_history(self) -> None:
        seeds = self.pipeline.state.seeds
        if seeds is not None:
            self._history.append(seeds.copy())
            self._future.clear()

    def add_seed(self) -> None:
        points = self.pipeline.state.points
        seeds = self.pipeline.state.seeds
        if points is None:
            QMessageBox.warning(self, "无法添加", "请先加载点云。")
            return
        self._push_seed_history()
        if seeds is not None and self.selected_seed is not None and 0 <= self.selected_seed < seeds.shape[0]:
            new_seed = seeds[self.selected_seed].copy()
            new_seed[2] += max(self.pipeline.config.grid_step, 0.02)
            seeds = np.vstack([seeds, new_seed])
            self.selected_seed = seeds.shape[0] - 1
        elif seeds is not None:
            seeds = np.vstack([seeds, points.mean(axis=0)])
            self.selected_seed = seeds.shape[0] - 1
        else:
            seeds = points.mean(axis=0, keepdims=True)
            self.selected_seed = 0
        self.pipeline.set_seeds(seeds)
        self.update_seed_table()
        self.update_coord_spins()
        self.update_plot()

    def delete_selected_seed(self) -> None:
        self._delete_seed_at(self.selected_seed)

    def _delete_seed_at(self, index: int | None) -> None:
        seeds = self.pipeline.state.seeds
        if seeds is None or index is None or not (0 <= index < seeds.shape[0]):
            return
        if seeds.shape[0] <= 1:
            QMessageBox.warning(self, "无法删除", "至少需要保留一个种子点。")
            return
        self._push_seed_history()
        seeds = np.delete(seeds, index, axis=0)
        self.selected_seed = min(index, seeds.shape[0] - 1)
        self.pipeline.set_seeds(seeds)
        self.update_seed_table()
        self.update_coord_spins()
        self.update_plot()

    def snap_selected_seed(self) -> None:
        points = self.pipeline.state.points
        seeds = self.pipeline.state.seeds
        if points is None or seeds is None or self.selected_seed is None:
            return
        self._push_seed_history()
        tree = cKDTree(points)
        _, idx = tree.query(seeds[self.selected_seed], k=1)
        seeds = seeds.copy()
        seeds[self.selected_seed] = points[int(idx)]
        self.pipeline.set_seeds(seeds)
        self.update_seed_table()
        self.update_coord_spins()
        self.update_plot()

    def undo_seed_edit(self) -> None:
        if not self._history:
            return
        current = self.pipeline.state.seeds
        if current is not None:
            self._future.append(current.copy())
        seeds = self._history.pop()
        self.selected_seed = min(self.selected_seed or 0, seeds.shape[0] - 1)
        self.pipeline.set_seeds(seeds)
        self.update_seed_table()
        self.update_coord_spins()
        self.update_plot()

    def redo_seed_edit(self) -> None:
        if not self._future:
            return
        current = self.pipeline.state.seeds
        if current is not None:
            self._history.append(current.copy())
        seeds = self._future.pop()
        self.selected_seed = min(self.selected_seed or 0, seeds.shape[0] - 1)
        self.pipeline.set_seeds(seeds)
        self.update_seed_table()
        self.update_coord_spins()
        self.update_plot()

    def _table_selection_changed(self) -> None:
        if self._updating_table:
            return
        rows = self.seed_table.selectionModel().selectedRows()
        if not rows:
            return
        self.selected_seed = int(rows[0].row())
        self.update_coord_spins()
        self.update_plot()

    def _seed_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item.column() not in (1, 2, 3):
            return
        seeds = self.pipeline.state.seeds
        if seeds is None:
            return
        row = item.row()
        if not (0 <= row < seeds.shape[0]):
            return
        try:
            value = float(item.text())
        except ValueError:
            self.update_seed_table()
            return
        self._push_seed_history()
        seeds = seeds.copy()
        seeds[row, item.column() - 1] = value
        self.selected_seed = row
        self.pipeline.set_seeds(seeds)
        self.update_coord_spins()
        self.update_plot()

    def _coord_spin_changed(self, axis: int, value: float) -> None:
        if self._updating_spins:
            return
        seeds = self.pipeline.state.seeds
        if seeds is None or self.selected_seed is None:
            return
        self._push_seed_history()
        seeds = seeds.copy()
        seeds[self.selected_seed, axis] = value
        self.pipeline.set_seeds(seeds)
        self.update_seed_table()
        self.update_plot()

    def update_seed_table(self) -> None:
        seeds = self.pipeline.state.seeds
        self._updating_table = True
        try:
            self.seed_table.setRowCount(0 if seeds is None else seeds.shape[0])
            if seeds is None:
                return
            for row, seed in enumerate(seeds):
                id_item = QTableWidgetItem(str(row))
                id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.seed_table.setItem(row, 0, id_item)
                for col in range(3):
                    self.seed_table.setItem(row, col + 1, QTableWidgetItem(f"{seed[col]:.6f}"))
                delete_btn = QToolButton()
                delete_btn.setText("删")
                delete_btn.clicked.connect(lambda _checked=False, r=row: self._delete_seed_at(r))
                self.seed_table.setCellWidget(row, 4, delete_btn)
            if self.selected_seed is not None and 0 <= self.selected_seed < self.seed_table.rowCount():
                self.seed_table.selectRow(self.selected_seed)
        finally:
            self._updating_table = False

    def update_coord_spins(self) -> None:
        seeds = self.pipeline.state.seeds
        self._updating_spins = True
        try:
            enabled = seeds is not None and self.selected_seed is not None and 0 <= self.selected_seed < seeds.shape[0]
            for spin in (self.x_spin, self.y_spin, self.z_spin):
                spin.setEnabled(enabled)
            if enabled:
                seed = seeds[self.selected_seed]
                self.x_spin.setValue(float(seed[0]))
                self.y_spin.setValue(float(seed[1]))
                self.z_spin.setValue(float(seed[2]))
        finally:
            self._updating_spins = False

    def _on_pick(self, event) -> None:
        if event.artist is not self.seed_scatter or not len(event.ind):
            return
        self.selected_seed = int(event.ind[0])
        self.update_seed_table()
        self.update_coord_spins()
        self.update_plot()

    def _on_mouse_press(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        if event.button in (2, 3):
            self._panning = True
            self._pan_last = (float(event.x), float(event.y))
            return
        if event.button != 1:
            return
        if self.drag_check.isChecked() and self.selected_seed is not None:
            self._dragging = True
            self._drag_last = (float(event.x), float(event.y))
            self._push_seed_history()
        else:
            self._rotating = True
            self._rotate_last = (float(event.x), float(event.y))

    def _on_mouse_move(self, event) -> None:
        if self._panning and self._pan_last is not None:
            self._pan_view(event)
            return
        if self._rotating and self._rotate_last is not None:
            self._rotate_view(event)
            return
        if not self._dragging or self._drag_last is None or self.selected_seed is None:
            return
        seeds = self.pipeline.state.seeds
        if seeds is None:
            return
        dx = float(event.x) - self._drag_last[0]
        dy = float(event.y) - self._drag_last[1]
        self._drag_last = (float(event.x), float(event.y))
        bbox = self.ax.bbox
        xrange = max(abs(self.ax.get_xlim3d()[1] - self.ax.get_xlim3d()[0]), 1e-9)
        yrange = max(abs(self.ax.get_ylim3d()[1] - self.ax.get_ylim3d()[0]), 1e-9)
        zrange = max(abs(self.ax.get_zlim3d()[1] - self.ax.get_zlim3d()[0]), 1e-9)
        sx = dx / max(float(bbox.width), 1.0)
        sy = dy / max(float(bbox.height), 1.0)
        plane = self.drag_plane.currentText()
        delta = np.zeros(3, dtype=float)
        if plane == "XY":
            delta = np.array([sx * xrange, -sy * yrange, 0.0])
        elif plane == "XZ":
            delta = np.array([sx * xrange, 0.0, -sy * zrange])
        else:
            delta = np.array([0.0, sx * yrange, -sy * zrange])
        seeds = seeds.copy()
        candidate = seeds[self.selected_seed] + delta
        seeds[self.selected_seed] = self._correct_drag_candidate(candidate)
        self.pipeline.set_seeds(seeds)
        self.update_seed_table()
        self.update_coord_spins()
        self.update_plot()

    def _on_mouse_release(self, event) -> None:
        _ = event
        self._dragging = False
        self._drag_last = None
        self._panning = False
        self._pan_last = None
        self._rotating = False
        self._rotate_last = None

    def _on_scroll_zoom(self, event) -> None:
        if event.inaxes is not self.ax:
            return
        scale = 0.82 if event.button == "up" else 1.22
        for getter, setter in (
            (self.ax.get_xlim3d, self.ax.set_xlim3d),
            (self.ax.get_ylim3d, self.ax.set_ylim3d),
            (self.ax.get_zlim3d, self.ax.set_zlim3d),
        ):
            lo, hi = getter()
            center = (lo + hi) / 2.0
            half = (hi - lo) * scale / 2.0
            setter(center - half, center + half)
        self._remember_axes_limits()
        self.canvas.draw_idle()

    def _pan_view(self, event) -> None:
        if event.x is None or event.y is None or self._pan_last is None:
            return
        dx = float(event.x) - self._pan_last[0]
        dy = float(event.y) - self._pan_last[1]
        self._pan_last = (float(event.x), float(event.y))
        bbox = self.ax.bbox
        x0, x1 = self.ax.get_xlim3d()
        y0, y1 = self.ax.get_ylim3d()
        sx = (x1 - x0) * dx / max(float(bbox.width), 1.0)
        sy = (y1 - y0) * dy / max(float(bbox.height), 1.0)
        self.ax.set_xlim3d(x0 - sx, x1 - sx)
        self.ax.set_ylim3d(y0 + sy, y1 + sy)
        self._remember_axes_limits()
        self.canvas.draw_idle()

    def _rotate_view(self, event) -> None:
        if event.x is None or event.y is None or self._rotate_last is None:
            return
        dx = float(event.x) - self._rotate_last[0]
        dy = float(event.y) - self._rotate_last[1]
        self._rotate_last = (float(event.x), float(event.y))
        elev = float(getattr(self.ax, "elev", 30.0)) - dy * 0.35
        azim = float(getattr(self.ax, "azim", -60.0)) - dx * 0.35
        elev = float(np.clip(elev, -89.0, 89.0))
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def _remember_axes_limits(self) -> None:
        self._manual_axes_limits = (
            tuple(float(v) for v in self.ax.get_xlim3d()),
            tuple(float(v) for v in self.ax.get_ylim3d()),
            tuple(float(v) for v in self.ax.get_zlim3d()),
        )

    def _apply_axes_limits(self) -> None:
        if self._manual_axes_limits is None:
            return
        self.ax.set_xlim3d(*self._manual_axes_limits[0])
        self.ax.set_ylim3d(*self._manual_axes_limits[1])
        self.ax.set_zlim3d(*self._manual_axes_limits[2])
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    def _correct_drag_candidate(self, candidate: np.ndarray) -> np.ndarray:
        points = self.pipeline.state.points
        mode = self.drag_mode.currentText()
        if points is None or mode == "平面自由":
            return candidate
        tree = cKDTree(points)
        if mode == "最近点吸附":
            _, idx = tree.query(candidate, k=1)
            return points[int(idx)]
        k = min(24, points.shape[0])
        _, idx = tree.query(candidate, k=k)
        idx = np.asarray(idx, dtype=int).reshape(-1)
        return points[idx].mean(axis=0)

    def update_plot(self, *_args) -> None:
        elev = float(getattr(self.ax, "elev", 30.0))
        azim = float(getattr(self.ax, "azim", -60.0))
        roll = float(getattr(self.ax, "roll", 0.0))
        self.ax.clear()
        state = self.pipeline.state
        points = state.points
        if points is None:
            self.canvas.draw_idle()
            return

        stage = self.stage_combo.currentIndex()
        self.seed_scatter = None

        point_size = float(self.point_size_spin.value())
        point_alpha = float(self.point_alpha_spin.value())
        max_points = int(self.max_points_spin.value())
        if self.point_visible_check.isChecked():
            label_view = stage == 2 or self.show_labels_check.isChecked()
            if label_view and state.labels is not None:
                self._draw_point_cloud(points, labels=state.labels, size=point_size, alpha=point_alpha, max_points=max_points)
            elif stage in (1, 3):
                self._draw_point_cloud(points, color="0.82", size=point_size, alpha=point_alpha, max_points=min(max_points, 3000))
            else:
                self._draw_point_cloud(points, color="0.45", size=point_size, alpha=point_alpha, max_points=max_points)

        if stage in (1, 3) and state.growth_vectors is not None:
            self._draw_growth_vectors(points, state.growth_vectors)

        if state.skeleton is not None and state.skeleton.stream_points.size:
            if stage == 3:
                self._draw_streamlines(state.skeleton.stream_points, color_by_label=True, linewidth=1.8, alpha=0.9)
            elif stage == 4 or self.show_stream_check.isChecked():
                self._draw_streamlines(state.skeleton.stream_points, color_by_label=True, linewidth=1.2, alpha=0.85)

        if state.skeleton is not None and state.skeleton.raw_edges.size and stage in (4, 5):
            self._draw_edges(state.skeleton.stream_points[:, :3], state.skeleton.raw_edges, "0.18", 0.9 if stage == 4 else 2.2)

        if state.skeleton is not None and state.skeleton.smooth_points.size and stage == 6:
            self._draw_edges(state.skeleton.smooth_points[:, :3], state.skeleton.refined_edges, "tab:blue", 1.9)

        if (
            self.seed_visible_check.isChecked()
            and state.seeds is not None
            and state.seeds.size
        ):
            self._draw_seeds(state.seeds)

        if self.metric_overlay_check.isChecked() and self.metric_overlay is not None:
            self._draw_metric_overlay()

        if self._manual_axes_limits is None:
            self._set_equal_axes(points)
        else:
            self._apply_axes_limits()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.grid(False)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.set_edgecolor((1, 1, 1, 0))
            axis.pane.set_alpha(0.0)
        try:
            self.ax.view_init(elev=elev, azim=azim, roll=roll)
        except TypeError:
            self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def _label_palette(self, count: int) -> np.ndarray:
        rng = np.random.default_rng(18)
        h = np.linspace(0.0, 1.0, count + 1)[:-1]
        h = h[rng.permutation(count)]
        s = 0.70 + 0.22 * rng.random(count)
        v = 0.78 + 0.18 * rng.random(count)
        return 0.92 * self._hsv_to_rgb(np.column_stack([h, s, v])) + 0.08

    def _hsv_to_rgb(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        i = np.floor(h * 6).astype(int)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i % 6
        rgb = np.zeros((hsv.shape[0], 3), dtype=float)
        choices = [
            np.column_stack([v, t, p]),
            np.column_stack([q, v, p]),
            np.column_stack([p, v, t]),
            np.column_stack([p, q, v]),
            np.column_stack([t, p, v]),
            np.column_stack([v, p, q]),
        ]
        for k, vals in enumerate(choices):
            rgb[i == k] = vals[i == k]
        return rgb

    def _sample_indices(self, points: np.ndarray, max_points: int) -> np.ndarray:
        if points.shape[0] <= max_points:
            return np.arange(points.shape[0])
        pick = np.round(np.linspace(0, points.shape[0] - 1, max_points)).astype(int)
        return np.unique(pick)

    def _draw_point_cloud(
        self,
        points: np.ndarray,
        labels: np.ndarray | None = None,
        color: str = "0.68",
        size: float = 5,
        alpha: float = 0.36,
        max_points: int | None = None,
    ) -> None:
        idx = np.arange(points.shape[0]) if max_points is None else self._sample_indices(points, max_points)
        mode = self.color_mode_combo.currentText()
        cmap = None
        vmin = None
        vmax = None
        if mode == "阶段自动" and labels is not None:
            labs = labels.astype(int)
            palette = self._label_palette(max(int(labs.max()) + 1, 1))
            colors = palette[labs[idx]]
        elif mode == "标签颜色(计算)" and self.pipeline.state.labels is not None:
            labs = self.pipeline.state.labels.astype(int)
            palette = self._label_palette(max(int(labs.max()) + 1, 1))
            colors = palette[labs[idx]]
        elif mode == "热扩散颜色(计算)" and "heat" in self.pipeline.state.aux:
            colors, vmin, vmax = self._auto_scaled_values(self.pipeline.state.aux["heat"], idx)
            cmap = "inferno"
        elif mode == "生长强度颜色(计算)" and "growth_magnitude" in self.pipeline.state.aux:
            colors, vmin, vmax = self._auto_scaled_values(self.pipeline.state.aux["growth_magnitude"], idx)
            cmap = "plasma"
        elif mode == "Z高度自动调节":
            colors, vmin, vmax = self._auto_scaled_values(points[:, 2], idx)
            cmap = "viridis"
        else:
            colors = color
        p = points[idx]
        self.ax.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            s=size,
            c=colors,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
            depthshade=False,
        )

    def _auto_scaled_values(self, values: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, float, float]:
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.shape[0] != self.pipeline.state.points.shape[0]:
            return np.zeros(idx.shape[0]), 0.0, 1.0
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros(idx.shape[0]), 0.0, 1.0
        lo, hi = np.percentile(finite, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1.0
        return np.clip(values[idx], lo, hi), float(lo), float(hi)

    def _draw_growth_vectors(self, points: np.ndarray, vectors: np.ndarray) -> None:
        idx = self._sample_indices(points, 2300)
        p = points[idx]
        vec = vectors[idx].copy()
        norm = np.linalg.norm(vec, axis=1)
        keep = np.isfinite(norm) & (norm > 1e-12)
        if not np.any(keep):
            return
        p = p[keep]
        vec = vec[keep] / norm[keep, None]
        diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) + 1e-9)
        vec = vec * (0.018 * diag)
        self.ax.quiver(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            vec[:, 0],
            vec[:, 1],
            vec[:, 2],
            color=(0.36, 0.36, 0.36),
            linewidth=0.62,
            arrow_length_ratio=0.23,
        )

    def _draw_streamlines(
        self,
        stream_points: np.ndarray,
        color_by_label: bool = True,
        linewidth: float = 1.2,
        alpha: float = 0.85,
    ) -> None:
        labels = stream_points[:, 3].astype(int)
        palette = self._label_palette(max(int(labels.max()) + 1, 1))
        for lab in np.unique(labels):
            pts = stream_points[labels == lab, :3]
            if pts.shape[0] < 2:
                continue
            color = palette[lab] if color_by_label else "tab:green"
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=linewidth, alpha=alpha)

    def _draw_edges(self, nodes: np.ndarray, edges: np.ndarray, color, linewidth: float) -> None:
        for u, v in np.asarray(edges, dtype=int):
            if u >= nodes.shape[0] or v >= nodes.shape[0]:
                continue
            self.ax.plot(
                [nodes[u, 0], nodes[v, 0]],
                [nodes[u, 1], nodes[v, 1]],
                [nodes[u, 2], nodes[v, 2]],
                color=color,
                linewidth=linewidth,
                alpha=0.95,
            )

    def _draw_metric_overlay(self) -> None:
        kind, key, row = self.metric_overlay
        if kind == "branch" and row is not None:
            self._draw_branch_metric_overlay(key, row)
        elif kind == "summary":
            self._draw_summary_metric_overlay(key)

    def _draw_branch_metric_overlay(self, key: str, row: int) -> None:
        state = self.pipeline.state
        if state.skeleton is None or row >= len(self.metric_branches):
            return
        branch = self.metric_branches[row]
        path_text = str(branch.get("path_nodes", ""))
        path = [int(x) for x in path_text.split(";") if x.strip().isdigit()]
        nodes = state.skeleton.smooth_points[:, :3]
        path = [i for i in path if 0 <= i < nodes.shape[0]]
        if len(path) < 2:
            return
        pts = nodes[path]
        value = branch.get(key, "")
        label = f"{key}={self._fmt_metric(value)}"
        mid = pts[len(pts) // 2]
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="magenta", linewidth=4.2, alpha=0.98)
        self.ax.scatter(
            pts[[0, -1], 0],
            pts[[0, -1], 1],
            pts[[0, -1], 2],
            s=90,
            c=["lime", "orange"],
            depthshade=False,
        )

        if key == "chord_length":
            self.ax.plot(pts[[0, -1], 0], pts[[0, -1], 1], pts[[0, -1], 2], color="cyan", linewidth=3.0, alpha=0.92)
        elif key == "horizontal_projection":
            a = pts[0].copy()
            b = pts[-1].copy()
            b[2] = a[2]
            self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="cyan", linewidth=3.0, alpha=0.92)
        elif key in {"vertical_rise", "base_height", "tip_height"}:
            a = pts[0].copy()
            b = pts[-1].copy()
            b[:2] = a[:2]
            self.ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="crimson", linewidth=3.0, alpha=0.92)
        elif key in {"parent_angle_deg", "inclination_from_vertical_deg", "elevation_deg", "azimuth_deg"}:
            self._draw_branch_angle_overlay(key, row, pts, nodes)
        elif key in {"radius_start", "diameter_start", "radius_end", "diameter_end", "taper_rate"}:
            r0 = float(branch.get("radius_start", 0.0) or 0.0)
            r1 = float(branch.get("radius_end", 0.0) or 0.0)
            self._draw_horizontal_circle(float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2]), max(r0, 1e-6), "crimson")
            self._draw_horizontal_circle(float(pts[-1, 0]), float(pts[-1, 1]), float(pts[-1, 2]), max(r1, 1e-6), "orange")
        elif key in {"volume", "surface_area", "slenderness", "tortuosity"}:
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="purple", linewidth=6.0, alpha=0.45)
        elif key in {"start_node", "end_node"}:
            idx = 0 if key == "start_node" else -1
            self.ax.scatter([pts[idx, 0]], [pts[idx, 1]], [pts[idx, 2]], s=170, c="cyan", depthshade=False)

        self.ax.text(mid[0], mid[1], mid[2], label, color="black")

    def _draw_branch_angle_overlay(self, key: str, row: int, pts: np.ndarray, nodes: np.ndarray) -> None:
        branch = self.metric_branches[row]
        joint = pts[0]
        vec = pts[-1] - pts[0]
        self.ax.quiver(joint[0], joint[1], joint[2], *vec, color="magenta", length=0.75, normalize=True)
        if key == "parent_angle_deg":
            parent_id = int(branch.get("parent_branch_id", -1))
            if 0 <= parent_id < len(self.metric_branches):
                ppath = [
                    int(x)
                    for x in str(self.metric_branches[parent_id].get("path_nodes", "")).split(";")
                    if x.strip().isdigit()
                ]
                ppath = [i for i in ppath if 0 <= i < nodes.shape[0]]
                if len(ppath) >= 2:
                    ppts = nodes[ppath]
                    pvec = ppts[-1] - ppts[0]
                    self.ax.plot(ppts[:, 0], ppts[:, 1], ppts[:, 2], color="gold", linewidth=3.0, alpha=0.82)
                    self.ax.quiver(joint[0], joint[1], joint[2], *pvec, color="gold", length=0.75, normalize=True)
        elif key == "inclination_from_vertical_deg":
            self.ax.quiver(joint[0], joint[1], joint[2], 0.0, 0.0, 1.0, color="gold", length=0.75, normalize=True)
        elif key == "elevation_deg":
            horiz = np.array([vec[0], vec[1], 0.0], dtype=float)
            self.ax.quiver(joint[0], joint[1], joint[2], *horiz, color="gold", length=0.75, normalize=True)
        elif key == "azimuth_deg":
            self.ax.quiver(joint[0], joint[1], joint[2], 1.0, 0.0, 0.0, color="gold", length=0.75, normalize=True)

    def _draw_summary_metric_overlay(self, key: str) -> None:
        state = self.pipeline.state
        if state.points is None or not self.metric_summary:
            return
        points = state.points
        value = self.metric_summary.get(key, "")
        zmin = float(points[:, 2].min())
        zmax = float(points[:, 2].max())
        center_xy = np.median(points[:, :2], axis=0)
        dbh_keys = {
            "dbh",
            "dbh_radius",
            "dbh_height",
            "dbh_z",
            "dbh_center_x",
            "dbh_center_y",
            "dbh_sample_count",
            "basal_area",
        }
        if key in {"tree_height", "crown_length", "crown_base_height"} | dbh_keys:
            x, y = center_xy
            if key == "tree_height":
                self.ax.plot([x, x], [y, y], [zmin, zmax], color="crimson", linewidth=4)
                self.ax.text(x, y, zmax, f"Height={self._fmt_metric(value)}", color="crimson")
            elif key in dbh_keys:
                self._draw_dbh_overlay(key, value)
            else:
                z = float(self.metric_summary.get("crown_base_height", zmin))
                self.ax.plot([x, x], [y, y], [z, zmax], color="green", linewidth=4)
                self.ax.text(x, y, z, f"{key}={self._fmt_metric(value)}", color="green")
        elif key.startswith("crown_"):
            self._draw_crown_overlay()
            self.ax.text(center_xy[0], center_xy[1], zmax, f"{key}={self._fmt_metric(value)}", color="green")
        elif key in {"terminal_tip_count", "bifurcation_count"}:
            self._draw_node_count_overlay(key)
            self.ax.text2D(0.02, 0.96, f"{key}: {self._fmt_metric(value)}", transform=self.ax.transAxes, color="black")
        elif key in {
            "branch_count",
            "total_branch_length",
            "total_wood_volume_frustum",
            "total_surface_area_frustum",
            "max_topological_order",
            "max_strahler_order",
            "mean_parent_branch_angle_deg",
            "max_parent_branch_angle_deg",
        }:
            if state.skeleton is not None and state.skeleton.smooth_points.size and state.skeleton.refined_edges.size:
                self._draw_edges(state.skeleton.smooth_points[:, :3], state.skeleton.refined_edges, "magenta", 2.4)
            self.ax.text2D(0.02, 0.96, f"{key}: {self._fmt_metric(value)}", transform=self.ax.transAxes, color="black")
        else:
            self.ax.text2D(0.02, 0.96, f"{key}: {self._fmt_metric(value)}", transform=self.ax.transAxes, color="black")

    def _draw_horizontal_circle(self, x: float, y: float, z: float, r: float, color: str) -> None:
        t = np.linspace(0, 2 * np.pi, 120)
        self.ax.plot(x + r * np.cos(t), y + r * np.sin(t), np.full_like(t, z), color=color, linewidth=3)

    def _draw_dbh_overlay(self, key: str, value) -> None:
        points = self.pipeline.state.points
        if points is None:
            return
        x = float(self.metric_summary.get("dbh_center_x", np.median(points[:, 0])))
        y = float(self.metric_summary.get("dbh_center_y", np.median(points[:, 1])))
        z = float(self.metric_summary.get("dbh_z", points[:, 2].min() + self.metric_summary.get("dbh_height", 1.3)))
        r = float(self.metric_summary.get("dbh_radius", 0.1))
        height = float(points[:, 2].max() - points[:, 2].min())
        band = max(0.18, 0.025 * height)
        mask = np.abs(points[:, 2] - z) <= band
        if np.count_nonzero(mask):
            p = points[mask]
            self.ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=18, c="gold", alpha=0.62, depthshade=False)
        self._draw_horizontal_circle(x, y, z, r, "crimson")
        self._draw_horizontal_circle(x, y, z, r * 0.5, "crimson")
        self.ax.scatter([x], [y], [z], s=130, c="cyan", edgecolors="black", depthshade=False)
        self.ax.plot([x - r, x + r], [y, y], [z, z], color="crimson", linewidth=3)
        self.ax.plot([x, x], [y - r, y + r], [z, z], color="crimson", linewidth=3)
        self.ax.text(x + r, y, z, f"{key}={self._fmt_metric(value)}", color="crimson")

    def _draw_crown_overlay(self) -> None:
        points = self.pipeline.state.points
        if points is None:
            return
        z0 = float(self.metric_summary.get("crown_base_height", points[:, 2].min()))
        z1 = float(points[:, 2].max())
        x0, y0 = points[:, :2].min(axis=0)
        x1, y1 = points[:, :2].max(axis=0)
        for z in (z0, z1):
            self.ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [z] * 5, color="green", linewidth=2.2)
        for x in (x0, x1):
            for y in (y0, y1):
                self.ax.plot([x, x], [y, y], [z0, z1], color="green", linewidth=1.5, alpha=0.7)

    def _draw_node_count_overlay(self, key: str) -> None:
        skeleton = self.pipeline.state.skeleton
        if skeleton is None or skeleton.smooth_points.size == 0 or skeleton.refined_edges.size == 0:
            return
        nodes = skeleton.smooth_points[:, :3]
        degree = np.zeros(nodes.shape[0], dtype=int)
        for u, v in np.asarray(skeleton.refined_edges, dtype=int):
            if 0 <= u < nodes.shape[0] and 0 <= v < nodes.shape[0]:
                degree[u] += 1
                degree[v] += 1
        root = int(np.argmin(nodes[:, 2]))
        if key == "terminal_tip_count":
            idx = np.flatnonzero((degree <= 1) & (np.arange(nodes.shape[0]) != root))
            color = "orange"
        else:
            idx = np.flatnonzero(degree >= 3)
            color = "cyan"
        if idx.size:
            p = nodes[idx]
            self.ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=110, c=color, edgecolors="black", depthshade=False)

    def _draw_seeds(self, seeds: np.ndarray) -> None:
        seed_color = self._seed_colors(seeds.shape[0])
        seed_size = float(self.seed_size_spin.value())
        self.seed_scatter = self.ax.scatter(
            seeds[:, 0],
            seeds[:, 1],
            seeds[:, 2],
            s=seed_size,
            c=seed_color,
            edgecolors="black",
            linewidths=0.65,
            depthshade=False,
            picker=8,
        )
        if self.selected_seed is not None and 0 <= self.selected_seed < seeds.shape[0]:
            seed = seeds[self.selected_seed]
            self.ax.scatter(
                [seed[0]],
                [seed[1]],
                [seed[2]],
                s=max(seed_size * 2.6, seed_size + 25.0),
                c="yellow",
                edgecolors="black",
                linewidths=1.0,
                depthshade=False,
            )

    def _seed_colors(self, count: int):
        mode = self.seed_color_combo.currentText()
        fixed = {
            "红色": "red",
            "蓝色": "tab:blue",
            "绿色": "tab:green",
            "黑色": "black",
            "白色": "white",
        }
        if mode in fixed:
            return fixed[mode]
        return self._label_palette(max(count, 1))[:count]

    def _set_equal_axes(self, points: np.ndarray) -> None:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        if self.pipeline.state.seeds is not None:
            mins = np.minimum(mins, self.pipeline.state.seeds.min(axis=0))
            maxs = np.maximum(maxs, self.pipeline.state.seeds.max(axis=0))
        center = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1e-6)
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    def _refresh_counts(self) -> None:
        state = self.pipeline.state
        p = 0 if state.points is None else state.points.shape[0]
        s = 0 if state.seeds is None else state.seeds.shape[0]
        n = 0 if state.skeleton is None else state.skeleton.smooth_points.shape[0]
        e = 0 if state.skeleton is None else state.skeleton.refined_edges.shape[0]
        self.statusBar().showMessage(f"点云 {p} | 种子 {s} | 骨架节点 {n} | 边 {e}")
