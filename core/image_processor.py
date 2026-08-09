"""图像加载、预处理与降采样到 24×24。

采样模式（mode 参数）：
- classic  : 直接 LANCZOS 重采样（旧行为，平均色，细线会被抹平）
- balanced : 主色采样（推荐默认）——先中心裁剪 1:1 消除变形，再对比度增强 +
             Unsharp 锐化保住边缘，最后分块量化取每格主色而非均值
- outline  : 线稿描边——在 balanced 基础上叠加 Canny 边缘检测，
             边缘格强制取深色，内部用主色，适合线稿/简笔画/图标
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter

_MODES = ("classic", "balanced", "outline")


def _center_crop_square(img: Image.Image) -> Image.Image:
    """按 1:1 中心裁剪，消除非方形图直接压成方形时的变形。"""
    w, h = img.size
    if w == h:
        return img
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def _auto_contrast(arr: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """亮度域自动色阶（保守版）。

    用灰度亮度计算 [P2, P98] 区间，对标量增益做标量拉伸，中心不变；
    增益封顶 2.5 防止过曝。全程用标量而非逐通道，不会破坏色相/饱和度
    （逐通道百分位拉伸会把纯色块的暗通道压到 0，毁掉饱和色）。
    """
    if amount <= 0:
        return arr
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    lo = float(np.percentile(gray, 2))
    hi = float(np.percentile(gray, 98))
    if hi - lo < 1e-6:
        return arr
    center = (lo + hi) / 2.0
    gain = min(255.0 / (hi - lo), 2.5)
    out = (arr - center) * gain + center
    return arr * (1.0 - amount) + out * amount


def _sharpen(img: Image.Image, amount: float = 0.8) -> Image.Image:
    """Unsharp Mask 锐化。必须在降采样之前做，让边缘在低分辨率下更清晰。"""
    if amount <= 0:
        return img
    return img.filter(ImageFilter.UnsharpMask(
        radius=2, percent=int(amount * 100), threshold=2))


def _dominant_color(pixels: np.ndarray) -> np.ndarray:
    """取一簇像素的代表色：量化到 5bit/通道统计色桶。

    三种情况：
    1. 块内呈现"背景+前景"双色结构（第二主色占比 ≥20%，如白底黑线、
       白底彩块）→ 取两者中更暗的一侧作为前景色，保住线条/主体色；
    2. 单一主色（占比 ≥15%）→ 直接取主色（均值会把黑白混成灰）；
    3. 无优势色（照片纹理）→ 退化均值，避免引入噪声。
    """
    q = (pixels // 8).astype(np.int16)               # 5bit 量化
    keys = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]   # 桶编号
    vals, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    c1 = int(counts[order[0]])
    c2 = int(counts[order[1]]) if len(counts) > 1 else 0
    n = len(pixels)
    m1 = pixels[keys == vals[order[0]]].mean(axis=0)

    if c2 >= max(2, int(n * 0.2)):
        # 双色结构：取更暗的一侧（前景线/前景色块通常比底暗）
        m2 = pixels[keys == vals[order[1]]].mean(axis=0)
        lum1 = float(m1 @ np.array([0.299, 0.587, 0.114]))
        lum2 = float(m2 @ np.array([0.299, 0.587, 0.114]))
        return m1 if lum1 < lum2 else m2
    if c1 >= max(2, int(n * 0.15)):
        return m1
    return pixels.mean(axis=0)


def _majority_sampling(img: Image.Image, size: int = 24,
                       intermediate: int = 240) -> np.ndarray:
    """主色采样：先缩到 intermediate²（每格对应 10×10 像素，统计有意义），
    再对每个 size×size 块取主色，得到 size² 输出。"""
    img = img.resize((intermediate, intermediate), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    block = intermediate // size
    out = np.zeros((size, size, 3), dtype=np.float64)
    for r in range(size):
        for c in range(size):
            blk = arr[r * block:(r + 1) * block,
                      c * block:(c + 1) * block].reshape(-1, 3)
            out[r, c] = _dominant_color(blk)
    return out


def _outline_mask(img: Image.Image, size: int,
                  edge_thresh: float = 0.22) -> np.ndarray:
    """Canny 边缘检测 → 每个 24×24 格内边缘像素占比超过阈值 → 该格是边缘格。

    边缘格将由调用方强制填深色，形成"描边像素画"效果。
    """
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    edges = cv2.Canny(gray, 60, 160)
    h, w = edges.shape
    bh, bw = h // size, w // size
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            blk = edges[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            mask[r, c] = float((blk > 0).mean()) > edge_thresh
    return mask


def load_and_downsample(path: str, size: int = 24, mode: str = "balanced",
                        sharpen: float = 0.8,
                        contrast: float = 1.0) -> np.ndarray:
    """读取图片并降采样到 size×size，返回 (size, size, 3) uint8 RGB。

    mode 取值见模块 docstring。classic 完全保持旧行为（直接 LANCZOS，
    不经任何预处理），用于回退对照；预处理只作用于 balanced/outline。
    """
    if mode not in _MODES:
        mode = "balanced"

    if mode == "classic":
        img = Image.open(path).convert("RGB")
        return np.asarray(img.resize((size, size), Image.LANCZOS),
                          dtype=np.uint8)

    img = Image.open(path).convert("RGB")
    img = _center_crop_square(img)

    arr = np.asarray(img, dtype=np.float64)
    arr = _auto_contrast(arr, amount=contrast)
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = _sharpen(img, amount=sharpen)

    out = _majority_sampling(img, size)
    if mode == "outline":
        mask = _outline_mask(img, size)
        out[mask] = (20, 20, 20)  # 深灰，匹配时归入色板黑色系
    return np.clip(out, 0, 255).astype(np.uint8)
