"""在 ROI 截图中定位 24×24 画布网格。"""
from __future__ import annotations

import numpy as np

from utils.opencv_helper import find_bright_canvas


class GridGeometry:
    """24×24 网格的屏幕几何信息（绝对坐标）。"""

    def __init__(self, origin_x: float, origin_y: float,
                 cell_w: float, cell_h: float, size: int = 24) -> None:
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.size = size

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        return (self.origin_x + (col + 0.5) * self.cell_w,
                self.origin_y + (row + 0.5) * self.cell_h)


def locate_grid(roi_bgr: np.ndarray, roi, size: int = 24,
                inset_ratio: float = 0.006) -> GridGeometry | None:
    """定位画布：找最大亮白矩形 → 内缩一点避开边框 → 等分 24×24。

    返回 GridGeometry（屏幕绝对坐标），失败返回 None。
    """
    rect = find_bright_canvas(roi_bgr)
    if rect is None:
        return None
    x, y, w, h = rect
    ix, iy = w * inset_ratio, h * inset_ratio
    x, y = x + ix, y + iy
    w, h = w - 2 * ix, h - 2 * iy
    return GridGeometry(roi.x + x, roi.y + y, w / size, h / size, size)
