"""把 24×24 像素矩阵匹配到已知色板（CIEDE2000 最近邻）。"""
from __future__ import annotations

import numpy as np

from core.color_math import delta_e_2000, rgb_to_lab


class MatchResult:
    def __init__(self, indices: np.ndarray, distances: np.ndarray) -> None:
        self.indices = indices      # (H, W) 每格匹配到的色板索引
        self.distances = distances  # (H, W) 每格的 CIEDE2000 色差


class PaletteMatcher:
    def __init__(self, palette_rgb: np.ndarray, threshold: float = 12.0,
                 luminance_weight: float = 1.2,
                 protect_gray: bool = True) -> None:
        self.palette_rgb = palette_rgb
        self.palette_lab = rgb_to_lab(palette_rgb)
        self.threshold = threshold
        # 像素画观感对明暗最敏感：选色时给亮度差加权，
        # 避免"色相对了但明暗错了"的失真感
        self.luminance_weight = luminance_weight
        # 灰度保护：色板在 L*≈30~65 区间没有灰色档，中灰像素若自由匹配
        # 会被"亮度接近的低色度褐色"抢走。低色度像素只在灰系色板色中选。
        self.protect_gray = protect_gray
        pal_chroma = np.hypot(self.palette_lab[:, 1], self.palette_lab[:, 2])
        self.gray_indices = np.where(pal_chroma < 12.0)[0]
        self.gray_chroma_limit = 10.0  # 像素色度低于此值视为"灰"

    def match(self, pixels: np.ndarray) -> MatchResult:
        """pixels: (H, W, 3) uint8 RGB → 每格最近的色板索引与色差。

        选色用 加权距离 = ΔE2000 + 亮度权重 × |ΔL*|；
        distances 保留纯 ΔE2000，用于阈值红框警示（语义不变）。
        全向量化：(576, 1, 3) 广播 (1, 40, 3)，比逐像素循环快约 20 倍。
        """
        h, w = pixels.shape[:2]
        labs = rgb_to_lab(pixels.reshape(-1, 3).astype(np.float64))
        d = delta_e_2000(labs[:, None, :], self.palette_lab[None, :, :])
        d_eff = d + self.luminance_weight * np.abs(
            labs[:, [0]] - self.palette_lab[None, :, 0])
        idx = np.argmin(d_eff, axis=1)

        if self.protect_gray and len(self.gray_indices) > 0:
            # 低色度像素 → 只在灰系色板色内按加权距离选
            pix_chroma = np.hypot(labs[:, 1], labs[:, 2])
            gray_mask = pix_chroma < self.gray_chroma_limit
            if gray_mask.any():
                d_gray = d_eff[np.ix_(gray_mask, self.gray_indices)]
                idx[gray_mask] = self.gray_indices[np.argmin(d_gray, axis=1)]

        dist = d[np.arange(len(labs)), idx]
        return MatchResult(idx.reshape(h, w), dist.reshape(h, w))

    def needed_colors(self, result: MatchResult) -> list[tuple[int, int]]:
        """统计每种色板颜色的用量 → [(palette_idx, count)] 按用量降序。"""
        flat = result.indices.reshape(-1)
        uniq, counts = np.unique(flat, return_counts=True)
        order = np.argsort(-counts)
        return [(int(uniq[i]), int(counts[i])) for i in order]
