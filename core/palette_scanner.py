"""在游戏 ROI 截图中识别当前可见的色板色块，映射到已知色板索引。

扫描范围限定在 ROI 右下区域（排除顶部工具栏）。为避免把画布格子、
按钮等误识别为色块，先定位深色色板面板的范围，只接受面板内的候选，
再按尺寸一致性过滤（同一色板的色块等大，排除滚动指示箭头等）。
"""
from __future__ import annotations

import cv2
import numpy as np

from core.color_math import delta_e_2000, rgb_to_lab
from core.palette_data import KnownPalette
from utils.opencv_helper import estimate_background, find_swatch_boxes

# 扫描区域（相对 ROI）：右侧 45%，纵向跳过顶部 15% 的工具栏
_RIGHT_RATIO = 0.55
_TOP_SKIP_RATIO = 0.15
# 面板暗色容差与面板外扩边距
_PANEL_TOL = 14
_PANEL_MARGIN = 8


class PaletteScanner:
    def __init__(self, palette: KnownPalette, roi, threshold: float = 10.0) -> None:
        self.palette = palette
        self.roi = roi
        self.threshold = threshold
        self.palette_lab = rgb_to_lab(palette.rgb)

    def scan(self, roi_bgr: np.ndarray) -> dict[int, tuple[int, int]]:
        """返回 {已知色板索引: (屏幕绝对 x, y)}，仅包含当前可见的色块。"""
        h, w = roi_bgr.shape[:2]
        x0 = int(w * _RIGHT_RATIO)
        y0 = int(h * _TOP_SKIP_RATIO)
        area = roi_bgr[y0:, x0:]

        panel = self._find_panel_rect(area)
        boxes = find_swatch_boxes(area)
        if panel is not None:
            boxes = [b for b in boxes if self._inside(panel, b[0], b[1])]
        boxes = self._filter_consistent_size(boxes)

        result: dict[int, tuple[int, int]] = {}
        for cx, cy, bw, bh, rgb in boxes:
            lab = rgb_to_lab(np.array(rgb, dtype=np.float64).reshape(1, 3))[0]
            d = delta_e_2000(lab, self.palette_lab)
            idx = int(np.argmin(d))
            if d[idx] <= self.threshold and idx not in result:
                result[idx] = (self.roi.x + x0 + cx, self.roi.y + y0 + cy)
        return result

    @staticmethod
    def _find_panel_rect(area: np.ndarray):
        """定位色板面板：扫描区域内最大的"近背景暗色"连通域。

        返回 (x, y, w, h)（相对 area）或 None。
        """
        bg = estimate_background(area)
        dark = (np.abs(area.astype(np.int16) - bg.astype(np.int16))
                .max(axis=2) <= _PANEL_TOL).astype(np.uint8)
        n, _labels, stats, _cents = cv2.connectedComponentsWithStats(dark, 8)
        if n <= 1:
            return None
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h = (int(v) for v in stats[biggest, :4])
        return (x, y, w, h)

    @staticmethod
    def _inside(rect, px: int, py: int) -> bool:
        x, y, w, h = rect
        m = _PANEL_MARGIN
        return (x - m) <= px <= (x + w + m) and (y - m) <= py <= (y + h + m)

    @staticmethod
    def _filter_consistent_size(boxes, tol_ratio: float = 0.3):
        """只保留边长接近中位数的色块（同一色板的色块等大）。"""
        if len(boxes) < 4:
            return boxes
        sides = np.array([min(b[2], b[3]) for b in boxes], dtype=np.float64)
        med = np.median(sides)
        return [b for b in boxes
                if abs(min(b[2], b[3]) - med) <= med * tol_ratio]
