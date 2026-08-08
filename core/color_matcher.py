"""把 24×24 像素矩阵匹配到已知色板（CIEDE2000 最近邻）。"""
from __future__ import annotations

import numpy as np

from core.color_math import delta_e_2000, rgb_to_lab


class MatchResult:
    def __init__(self, indices: np.ndarray, distances: np.ndarray) -> None:
        self.indices = indices      # (H, W) 每格匹配到的色板索引
        self.distances = distances  # (H, W) 每格的 CIEDE2000 色差


class PaletteMatcher:
    def __init__(self, palette_rgb: np.ndarray, threshold: float = 12.0) -> None:
        self.palette_rgb = palette_rgb
        self.palette_lab = rgb_to_lab(palette_rgb)
        self.threshold = threshold

    def match(self, pixels: np.ndarray) -> MatchResult:
        """pixels: (H, W, 3) uint8 RGB → 每格最近的色板索引与色差。"""
        h, w = pixels.shape[:2]
        labs = rgb_to_lab(pixels.reshape(-1, 3).astype(np.float64))
        idx = np.empty(len(labs), dtype=np.int64)
        dist = np.empty(len(labs), dtype=np.float64)
        for i in range(len(labs)):
            d = delta_e_2000(labs[i], self.palette_lab)
            idx[i] = int(np.argmin(d))
            dist[i] = d[idx[i]]
        return MatchResult(idx.reshape(h, w), dist.reshape(h, w))

    def needed_colors(self, result: MatchResult) -> list[tuple[int, int]]:
        """统计每种色板颜色的用量 → [(palette_idx, count)] 按用量降序。"""
        flat = result.indices.reshape(-1)
        uniq, counts = np.unique(flat, return_counts=True)
        order = np.argsort(-counts)
        return [(int(uniq[i]), int(counts[i])) for i in order]
