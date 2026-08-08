"""图像加载与降采样到 24×24。"""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_and_downsample(path: str, size: int = 24) -> np.ndarray:
    """读取图片并降采样到 size×size，返回 (size, size, 3) uint8 RGB。

    LANCZOS 重采样在缩小照片类图片时保留最好的平均色彩；
    后续由 color_matcher 映射到游戏色板。
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)
