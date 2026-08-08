"""加载已知色板 known_palette.json（由 tools/extract_palette.py 生成）。"""
from __future__ import annotations

import json
import os

import numpy as np

from config import PALETTE_PATH


class KnownPalette:
    """游戏完整色板：颜色 RGB + 名称 + 在滚动列表中的行列位置。"""

    def __init__(self, colors: list[dict], columns: int = 4) -> None:
        self.colors = colors
        self.columns = columns
        self.rgb = np.array([c["rgb"] for c in colors], dtype=np.float64)
        self.names = [c["name"] for c in colors]

    def __len__(self) -> int:
        return len(self.colors)

    def name(self, idx: int) -> str:
        return self.names[idx]

    @property
    def whitest_index(self) -> int:
        """色板中最接近纯白的颜色索引（用于"跳过白色格"选项）。"""
        return int(np.argmax(self.rgb.sum(axis=1)))

    @classmethod
    def load(cls, path: str = PALETTE_PATH) -> "KnownPalette | None":
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["colors"], data.get("grid_columns", 4))
