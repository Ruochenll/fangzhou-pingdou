"""在游戏 ROI 截图中识别当前可见的色板色块，映射到已知色板索引。

分辨率/布局自适应，不假设色板位置：
1. 全 ROI 检测所有实心方形候选 + 所有深色面板
2. 选"内部候选最多"的深色面板作为色板（色板在左/右/下均可）
3. 面板内候选按尺寸一致性过滤（同一色板的色块等大）
4. 色块中心颜色与已知色板做 CIEDE2000 匹配
"""
from __future__ import annotations

import numpy as np

from core.color_math import delta_e_2000, rgb_to_lab
from core.palette_data import KnownPalette
from utils.opencv_helper import find_dark_panels, find_swatch_boxes

# 面板外扩边距（候选中心允许超出面板边界这么多）
_PANEL_MARGIN = 8
# 一个面板至少包含这么多候选才被认为是色板
_MIN_SWATCHES_IN_PANEL = 4


class PaletteScanner:
    def __init__(self, palette: KnownPalette, roi, threshold: float = 10.0) -> None:
        self.palette = palette
        self.roi = roi
        self.threshold = threshold
        self.palette_lab = rgb_to_lab(palette.rgb)

    def scan(self, roi_bgr: np.ndarray) -> dict[int, tuple[int, int]]:
        """返回 {已知色板索引: (屏幕绝对 x, y)}，仅包含当前可见的色块。"""
        # 全 ROI 检测候选（不排除任何色系，白色色块也能检出）
        boxes = find_swatch_boxes(roi_bgr, min_side=10, max_side=300,
                                  bg_tol=None)
        # 选包含候选最多的深色面板作为色板
        panel, boxes = self._pick_palette_panel(roi_bgr, boxes)
        boxes = self._filter_consistent_size(boxes)

        result: dict[int, tuple[int, int]] = {}
        for cx, cy, bw, bh, rgb in boxes:
            lab = rgb_to_lab(np.array(rgb, dtype=np.float64).reshape(1, 3))[0]
            d = delta_e_2000(lab, self.palette_lab)
            idx = int(np.argmin(d))
            if d[idx] <= self.threshold and idx not in result:
                result[idx] = (self.roi.x + cx, self.roi.y + cy)
        return result

    @staticmethod
    def _inside(rect, px: int, py: int) -> bool:
        x, y, w, h = rect[:4]
        m = _PANEL_MARGIN
        return (x - m) <= px <= (x + w + m) and (y - m) <= py <= (y + h + m)

    def _pick_palette_panel(self, roi_bgr: np.ndarray, boxes):
        """在所有深色面板中，选内部方形候选最多的那个作为色板。

        返回 (panel_rect, 面板内的候选列表)。找不到面板时返回 (None, 全部候选)。
        """
        panels = find_dark_panels(roi_bgr)
        best_panel, best_boxes = None, []
        for p in panels:
            inside = [b for b in boxes if self._inside(p, b[0], b[1])]
            if len(inside) > len(best_boxes):
                best_panel, best_boxes = p, inside
        if best_panel is not None and len(best_boxes) >= _MIN_SWATCHES_IN_PANEL:
            return best_panel, best_boxes
        return None, boxes  # 兜底：找不到面板就用全部候选

    @staticmethod
    def _filter_consistent_size(boxes, tol_ratio: float = 0.3):
        """只保留边长接近中位数的色块（同一色板的色块等大）。"""
        if len(boxes) < 4:
            return boxes
        sides = np.array([min(b[2], b[3]) for b in boxes], dtype=np.float64)
        med = np.median(sides)
        return [b for b in boxes
                if abs(min(b[2], b[3]) - med) <= med * tol_ratio]
