"""在游戏 ROI 截图中识别当前可见的色板色块，映射到已知色板索引。

分辨率/布局自适应，不假设色板位置：
1. 全 ROI 检测所有实心方形候选 + 所有深色面板
2. 选"内部候选最多且列数少"的深色面板作为色板
   （色板只有 4 列；画布填充深色颜料后有 24 列格子候选，以此区分）
3. 定位成功后缓存面板矩形，后续滚动只在缓存面板内扫描
   （面板在屏幕上不动，只有内容滚动；避免画布填色后干扰定位）
4. 面板内候选按尺寸一致性过滤（同一色板的色块等大）
5. 色块中心颜色与已知色板做 CIEDE2000 匹配
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
# 缓存面板内至少识别到这么多色块，否则认为缓存失效、重新全量定位
_MIN_CACHED_HITS = 4
# 色板候选的横向列数上限（色板 4 列；画布 24 列，以此排除画布误判）
_MAX_PALETTE_COLUMNS = 6
# 缓存面板后，识别数达到此值才更新缓存
_MIN_HITS_TO_CACHE = 8


class PaletteScanner:
    def __init__(self, palette: KnownPalette, roi, threshold: float = 10.0) -> None:
        self.palette = palette
        self.roi = roi
        self.threshold = threshold
        self.palette_lab = rgb_to_lab(palette.rgb)
        self._panel_rect: tuple[int, int, int, int] | None = None  # 相对 ROI

    # ---------- 主入口 ----------

    def scan(self, roi_bgr: np.ndarray) -> dict[int, tuple[int, int]]:
        """返回 {已知色板索引: (屏幕绝对 x, y)}，仅包含当前可见的色块。"""
        if self._panel_rect is not None:
            boxes = self._scan_cached_panel(roi_bgr)
            if len(boxes) < _MIN_CACHED_HITS:
                self._panel_rect = None  # 缓存失效，回退全量定位
        if self._panel_rect is None:
            boxes = find_swatch_boxes(roi_bgr, min_side=10, max_side=300,
                                      bg_tol=None)
            panel, boxes = self._pick_palette_panel(roi_bgr, boxes)
            boxes = self._filter_consistent_size(boxes)
            if panel is not None and len(boxes) >= _MIN_HITS_TO_CACHE:
                self._panel_rect = panel[:4]

        result: dict[int, tuple[int, int]] = {}
        for cx, cy, bw, bh, rgb in boxes:
            lab = rgb_to_lab(np.array(rgb, dtype=np.float64).reshape(1, 3))[0]
            d = delta_e_2000(lab, self.palette_lab)
            idx = int(np.argmin(d))
            if d[idx] <= self.threshold and idx not in result:
                result[idx] = (self.roi.x + cx, self.roi.y + cy)
        return result

    # ---------- 内部 ----------

    def _scan_cached_panel(self, roi_bgr: np.ndarray):
        """只在缓存的色板面板区域内扫描（面板位置不动，仅内容滚动）。"""
        x, y, w, h = self._panel_rect
        m = _PANEL_MARGIN
        h_img, w_img = roi_bgr.shape[:2]
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, w_img), min(y + h + m, h_img)
        if x1 <= x0 or y1 <= y0:
            return []
        area = roi_bgr[y0:y1, x0:x1]
        boxes = find_swatch_boxes(area, min_side=10, max_side=300, bg_tol=None)
        boxes = [(cx + x0, cy + y0, bw, bh, rgb) for cx, cy, bw, bh, rgb in boxes]
        return self._filter_consistent_size(boxes)

    @staticmethod
    def _inside(rect, px: int, py: int) -> bool:
        x, y, w, h = rect[:4]
        m = _PANEL_MARGIN
        return (x - m) <= px <= (x + w + m) and (y - m) <= py <= (y + h + m)

    @staticmethod
    def _column_count(boxes) -> int:
        """候选的横向列数（按 cx 量化到 10px）。"""
        return len({round(b[0] / 10) for b in boxes})

    def _pick_palette_panel(self, roi_bgr: np.ndarray, boxes):
        """在所有深色面板中，选"内部候选最多且列数像色板"的那个。

        返回 (panel_rect, 面板内的候选列表)。找不到时返回 (None, 全部候选)。
        """
        panels = find_dark_panels(roi_bgr)
        best_panel, best_boxes = None, []
        for p in panels:
            inside = [b for b in boxes if self._inside(p, b[0], b[1])]
            if len(inside) < _MIN_SWATCHES_IN_PANEL:
                continue
            # 画布填色后会成为深色面板候选，但它有 24 列格子 → 排除
            if self._column_count(inside) > _MAX_PALETTE_COLUMNS:
                continue
            if len(inside) > len(best_boxes):
                best_panel, best_boxes = p, inside
        if best_panel is not None:
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
