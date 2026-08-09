"""主窗口：导入图片 → 预览与色块清单 → 框选游戏区域 → 检测界面 → 开始填充。"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QFileDialog, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSlider, QSplitter, QVBoxLayout,
                               QWidget)

from config import DEBUG_DIR, GRID_SIZE, AppConfig, Region
from core.color_matcher import PaletteMatcher
from core.grid_locator import locate_grid
from core.image_processor import load_and_downsample
from core.palette_data import KnownPalette
from core.palette_scanner import PaletteScanner
from core.painter import PaintWorker
from ui.image_view import PixelCanvasView
from ui.palette_view import NeededColorsView
from ui.region_selector import RegionSelector
from utils.input import click
from utils.screenshot import ScreenCapturer


class MainWindow(QMainWindow):
    hotkey_stop = Signal()  # ESC 全局热键 → 跨线程安全地转到主线程停止

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("拼豆像素画助手")
        self.resize(1020, 660)

        self.cfg = AppConfig.load()
        self.palette = KnownPalette.load()
        self.pixels: np.ndarray | None = None
        self.match = None
        self.worker: PaintWorker | None = None
        self.white_idx = -1
        self._selector: RegionSelector | None = None

        self._build_ui()
        self._setup_hotkey()

        if self.palette is None:
            self.log("⚠ 未找到 known_palette.json，请先运行："
                     "python tools/extract_palette.py <色板截图1> <色板截图2>")
            self.btn_start.setEnabled(False)
        else:
            self.log(f"已加载色板：{len(self.palette)} 色")
        if self.cfg.roi.is_valid():
            r = self.cfg.roi
            self.log(f"已恢复上次框选的游戏区域：{r.w}×{r.h} @ ({r.x}, {r.y})")

    # ---------- UI ----------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        bar = QHBoxLayout()
        self.btn_import = QPushButton("① 导入图片")
        self.btn_roi = QPushButton("② 框选游戏区域")
        self.btn_detect = QPushButton("③ 检测界面")
        self.btn_test = QPushButton("点击测试")
        self.btn_test.setToolTip("在框选区域中心点击一次，验证游戏能否接收到自动化点击")
        self.btn_start = QPushButton("④ 开始填充")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.chk_skip_white = QCheckBox("跳过纯白格（画布默认白底）")
        self.chk_skip_white.setChecked(True)
        bar.addWidget(self.chk_skip_white)

        bar.addWidget(QLabel(" 采样:"))
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("主色均衡（推荐）", "balanced")
        self.cmb_mode.addItem("线稿描边", "outline")
        self.cmb_mode.addItem("经典 LANCZOS", "classic")
        bar.addWidget(self.cmb_mode)
        bar.addWidget(QLabel("锐化:"))
        self.sld_sharpen = QSlider(Qt.Horizontal)
        self.sld_sharpen.setRange(0, 15)
        self.sld_sharpen.setValue(8)
        self.sld_sharpen.setFixedWidth(90)
        bar.addWidget(self.sld_sharpen)

        for b in (self.btn_import, self.btn_roi, self.btn_detect,
                  self.btn_test, self.btn_start, self.btn_stop):
            bar.addWidget(b)
        bar.addStretch(1)
        root.addLayout(bar)

        self.btn_import.clicked.connect(self.on_import)
        self.btn_roi.clicked.connect(self.on_pick_roi)
        self.btn_detect.clicked.connect(self.on_detect)
        self.btn_test.clicked.connect(self.on_test_click)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        # 采样参数变化 → 若已导入图片则重新采样预览
        self.cmb_mode.currentIndexChanged.connect(self._on_sampling_changed)
        self.sld_sharpen.valueChanged.connect(self._on_sampling_changed)

        split = QSplitter()
        left = QWidget()
        ll = QVBoxLayout(left)
        self.canvas = PixelCanvasView(GRID_SIZE)
        ll.addWidget(self.canvas)
        self.lbl_info = QLabel("未加载图片")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        ll.addWidget(self.lbl_info)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(QLabel("所需色块（按用量排序）："))
        self.colors_view = NeededColorsView()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.colors_view)
        scroll.setMaximumHeight(190)
        rl.addWidget(scroll)
        rl.addWidget(QLabel("日志："))
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        rl.addWidget(self.log_view)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split)

    def _setup_hotkey(self) -> None:
        self.hotkey_stop.connect(self.on_stop)
        try:
            import keyboard
            keyboard.on_press_key("esc", lambda e: self.hotkey_stop.emit())
            self._hotkey_ok = True
        except Exception:
            self._hotkey_ok = False

    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    # ---------- ① 导入图片 ----------
    def _sampling_mode(self) -> str:
        return str(self.cmb_mode.currentData()) if self.cmb_mode else "balanced"

    def _sharpen_amount(self) -> float:
        return self.sld_sharpen.value() / 10.0

    def _on_sampling_changed(self) -> None:
        """采样模式/锐化变化 → 用已导入图片重新采样并刷新预览。"""
        if self.pixels is None or not getattr(self, "_last_path", None):
            return
        self.pixels = load_and_downsample(
            self._last_path, GRID_SIZE,
            mode=self._sampling_mode(),
            sharpen=self._sharpen_amount())
        self.log(f"采样参数已更新：{self._sampling_mode()}，锐化 "
                 f"{self._sharpen_amount():.1f} → 重新预览")
        self._refresh_preview()

    def on_import(self) -> None:
        if self.palette is None:
            QMessageBox.warning(self, "缺少色板",
                                "请先运行 tools/extract_palette.py 提取色板。")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return

        self._last_path = path
        self.pixels = load_and_downsample(
            path, GRID_SIZE,
            mode=self._sampling_mode(),
            sharpen=self._sharpen_amount())
        self.log(f"已加载图片：{path}（采样 {self._sampling_mode()}，锐化 "
                 f"{self._sharpen_amount():.1f}）")
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        """用当前 pixels 刷新 24×24 预览 + 色块清单（导入与参数变化共用）。"""
        if self.pixels is None or self.palette is None:
            return
        matcher = PaletteMatcher(self.palette.rgb, self.cfg.match_threshold)
        self.match = matcher.match(self.pixels)
        self.white_idx = self.palette.whitest_index

        mapped = self.palette.rgb[self.match.indices].astype(np.uint8)
        warn = self.match.distances > self.cfg.match_threshold
        self.canvas.set_data(mapped, warn)

        needed = matcher.needed_colors(self.match)
        items = [(tuple(int(v) for v in self.palette.rgb[i]),
                  self.palette.name(i), n) for i, n in needed]
        self.colors_view.set_items(items)

        n_warn = int(warn.sum())
        self.lbl_info.setText(
            f"{os.path.basename(self._last_path)} → {len(needed)} 种颜色"
            + (f"，{n_warn} 格色差较大（红框，将用最近色）" if n_warn else ""))

    # ---------- ② 框选区域 ----------
    def on_pick_roi(self) -> None:
        self.showMinimized()
        QApplication.processEvents()
        time.sleep(0.3)
        self._selector = RegionSelector()
        self._selector.region_selected.connect(self._on_roi_selected)
        self._selector.cancelled.connect(self._on_roi_cancelled)
        self._selector.show()

    def _on_roi_selected(self, rect: QRect) -> None:
        self.cfg.roi = Region(rect.x(), rect.y(), rect.width(), rect.height())
        self.cfg.save()
        self.showNormal()
        self.log(f"游戏区域已保存：{rect.width()}×{rect.height()} @ "
                 f"({rect.x()}, {rect.y()})")

    def _on_roi_cancelled(self) -> None:
        self.showNormal()
        self.log("已取消框选")

    # ---------- ③ 检测界面 ----------
    def _grab_roi(self) -> np.ndarray:
        """最小化本窗口后截取游戏区域，再恢复窗口。"""
        self.showMinimized()
        QApplication.processEvents()
        time.sleep(0.5)
        img = ScreenCapturer().grab_region(self.cfg.roi)
        self.showNormal()
        return img

    def on_detect(self) -> None:
        if not self.cfg.roi.is_valid():
            QMessageBox.warning(self, "缺少区域", "请先点击「② 框选游戏区域」。")
            return
        if self.palette is None:
            QMessageBox.warning(self, "缺少色板",
                                "请先运行 tools/extract_palette.py 提取色板。")
            return

        img = self._grab_roi()
        roi = self.cfg.roi
        grid = locate_grid(img, roi)
        visible = PaletteScanner(self.palette, roi,
                                 self.cfg.swatch_threshold).scan(img)

        dbg = img.copy()
        if grid is not None:
            x0 = int(grid.origin_x - roi.x)
            y0 = int(grid.origin_y - roi.y)
            x1 = int(x0 + grid.cell_w * grid.size)
            y1 = int(y0 + grid.cell_h * grid.size)
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 200, 0), 2)
            for r in range(0, grid.size, 4):
                for c in range(0, grid.size, 4):
                    cx, cy = grid.cell_center(r, c)
                    cv2.circle(dbg, (int(cx - roi.x), int(cy - roi.y)),
                               2, (0, 150, 0), -1)
        for idx, (sx, sy) in visible.items():
            cv2.circle(dbg, (sx - roi.x, sy - roi.y), 10, (0, 255, 255), 2)
        os.makedirs(DEBUG_DIR, exist_ok=True)
        out = os.path.join(DEBUG_DIR, "detect_overlay.png")
        cv2.imwrite(out, dbg)

        if grid is None:
            self.log("✗ 画布定位失败：未找到白色画布，请调整框选范围")
        else:
            self.log(f"✓ 画布定位：格子 {grid.cell_w:.1f}×{grid.cell_h:.1f} px")
        names = "、".join(self.palette.name(i) for i in sorted(visible))
        self.log(f"✓ 当前可见色块 {len(visible)} 个：{names}")
        self.log(f"检测叠加图已保存：{out}")
        self._show_image_popup(dbg, "检测结果（绿框=画布，黄圈=识别到的色块）")

    def _show_image_popup(self, bgr: np.ndarray, title: str) -> None:
        h, w = bgr.shape[:2]
        qimg = QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888).copy()
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        lay = QVBoxLayout(dlg)
        lb = QLabel()
        pix = QPixmap.fromImage(qimg)
        if pix.width() > 900:
            pix = pix.scaledToWidth(900, Qt.SmoothTransformation)
        lb.setPixmap(pix)
        lay.addWidget(lb)
        dlg.exec()

    # ---------- 点击测试 ----------
    def on_test_click(self) -> None:
        if not self.cfg.roi.is_valid():
            QMessageBox.information(self, "提示", "请先点击「② 框选游戏区域」。")
            return
        self.showMinimized()
        QApplication.processEvents()
        time.sleep(0.5)
        r = self.cfg.roi
        cx, cy = r.x + r.w // 2, r.y + r.h // 2
        click(cx, cy, hold_ms=150)
        time.sleep(0.3)
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.log(f"已在区域中心 ({cx}, {cy}) 点击一次")
        self.log("请观察游戏：画布中心是否出现了一个色点？")
        self.log("  - 出现了 → 点击链路正常，可以直接开始填充")
        self.log("  - 没出现 → 游戏可能在拦截注入输入，见下述排查")

    # ---------- ④ 开始填充 ----------
    def on_start(self) -> None:
        if self.match is None:
            QMessageBox.warning(self, "缺少图片", "请先点击「① 导入图片」。")
            return
        if not self.cfg.roi.is_valid():
            QMessageBox.warning(self, "缺少区域", "请先点击「② 框选游戏区域」。")
            return

        plan: dict[int, list[tuple[int, int]]] = {}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                idx = int(self.match.indices[r, c])
                if self.chk_skip_white.isChecked() and idx == self.white_idx:
                    continue
                plan.setdefault(idx, []).append((r, c))

        skipped = GRID_SIZE * GRID_SIZE - sum(len(v) for v in plan.values())
        if skipped:
            self.log(f"跳过纯白格 {skipped} 个")

        self.showMinimized()
        self.worker = PaintWorker(plan, self.palette, self.cfg.roi, self.cfg)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self._on_progress)
        self.worker.cell_done.connect(
            lambda r, c, _idx: self.canvas.mark_painted(r, c))
        self.worker.finished_all.connect(self._on_finished_all)
        self.worker.finished_partial.connect(self._on_finished_partial)
        self.worker.failed.connect(self._on_failed)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        tip = "ESC 或点击「停止」可中断；鼠标推到屏幕角落也可紧急中断"
        self.log(tip if self._hotkey_ok else "提示：ESC 热键不可用，请用「停止」按钮中断")
        self.worker.start()

    def on_stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.log("正在停止…（当前点击完成后即停）")

    def _on_progress(self, done: int, total: int) -> None:
        self.lbl_info.setText(f"填充进度：{done} / {total}")

    def _finish_ui(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_finished_all(self) -> None:
        self._finish_ui()
        self.log("✓ 全部格子填充完成！请在游戏中检查并手动保存/发布。")

    def _on_finished_partial(self, remaining: int) -> None:
        self._finish_ui()
        self.log(f"结束：剩余 {remaining} 格未填（色板中找不到对应颜色或被中断），"
                 "可在游戏中手动补填。")

    def _on_failed(self, msg: str) -> None:
        self._finish_ui()
        self.log(f"✗ 失败：{msg}")
        QMessageBox.warning(self, "填充失败", msg)
