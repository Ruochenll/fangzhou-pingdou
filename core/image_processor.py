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

from core.color_math import delta_e_2000, rgb_to_lab

_MODES = ("classic", "balanced", "outline", "kmeans", "dither")


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


def _kmeans_quantize(img: Image.Image, k: int = 16) -> Image.Image:
    """全图 k-means 预量化到 k 色：消除照片纹理噪声，让后续采样时
    每格的"主体颜色"更明确。先缩到 ≤240px 使耗时与原图尺寸无关。
    返回 PIL Image。"""
    if max(img.size) > 240:
        img = img.resize((240, 240), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    Z = arr.reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 3,
                                    cv2.KMEANS_PP_CENTERS)
    q = centers[labels.flatten()].reshape(arr.shape)
    return Image.fromarray(np.clip(q, 0, 255).astype(np.uint8))


def _dither_to_palette(arr_rgb: np.ndarray,
                       palette_rgb: np.ndarray,
                       protect_gray: bool = True) -> np.ndarray:
    """Floyd-Steinberg 误差扩散：把 (H,W,3) 图量化到色板。

    每格的量化误差扩散到右/左下/下/右下相邻格，用 40 色板的空间
    组合模拟中间色——渐变与肤色在 24×24 下的经典解法。
    距离用 CIEDE2000 + 1.2×亮度加权，与色板匹配口径一致；
    protect_gray 与 PaletteMatcher 的灰度保护规则一致。
    """
    h, w = arr_rgb.shape[:2]
    pal = palette_rgb.astype(np.float64)
    pal_lab = rgb_to_lab(pal)
    pal_chroma = np.hypot(pal_lab[:, 1], pal_lab[:, 2])
    gray_idx = np.where(pal_chroma < 12.0)[0]
    # 灰度保护的判定必须用原始像素色度：扩散误差会把中间态像素
    # 推出低色度区，导致保护中途失效、又匹配回褐色
    orig_lab = rgb_to_lab(arr_rgb.reshape(-1, 3)).reshape(h, w, 3)
    orig_chroma = np.hypot(orig_lab[..., 1], orig_lab[..., 2])
    out = arr_rgb.astype(np.float64).copy()
    res = np.zeros((h, w, 3), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            # 误差扩散会让值越出 [0,255]，参与色板匹配前必须 clip，
            # 否则 rgb_to_lab 对负数取 2.4 次幂产生 NaN
            old = np.clip(out[y, x], 0, 255)
            old_lab = rgb_to_lab(old.reshape(1, 3))[0]
            d = delta_e_2000(old_lab, pal_lab)
            d_eff = d + 1.2 * np.abs(old_lab[0] - pal_lab[:, 0])
            if (protect_gray and len(gray_idx) > 0
                    and orig_chroma[y, x] < 10.0):
                idx = int(gray_idx[int(np.argmin(d_eff[gray_idx]))])
            else:
                idx = int(np.argmin(d_eff))
            res[y, x] = pal[idx]
            err = old - pal[idx]
            if x + 1 < w:
                out[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    out[y + 1, x - 1] += err * 3 / 16
                out[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    out[y + 1, x + 1] += err * 1 / 16
    return res


def load_and_downsample(path: str, size: int = 24, mode: str = "balanced",
                        sharpen: float = 0.8, contrast: float = 1.0,
                        palette_rgb: np.ndarray | None = None) -> np.ndarray:
    """读取图片并降采样到 size×size，返回 (size, size, 3) uint8 RGB。

    mode 取值见模块 docstring。classic 完全保持旧行为（直接 LANCZOS，
    不经任何预处理），用于回退对照；dither 模式必须传 palette_rgb。
    需要反复调参重采样时，请用 ImageSampler（带预处理缓存）。
    """
    return ImageSampler().sample(path, size, mode, sharpen, contrast,
                                 palette_rgb)


class ImageSampler:
    """带预处理缓存的采样器。

    同一张图反复调参（锐化/模式）时，读图 + 裁剪 + 对比度只算一次；
    预处理中间图限制在 480px，各阶段耗时与原图尺寸无关
    （majority 缩到 240、kmeans 缩到 240、dither 直接用 24）。
    """

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._base: Image.Image | None = None

    def _preprocessed(self, path: str, contrast: float) -> Image.Image:
        key = (path, contrast)
        if key != self._key or self._base is None:
            img = Image.open(path).convert("RGB")
            img = _center_crop_square(img)
            if max(img.size) > 480:
                img = img.resize((480, 480), Image.LANCZOS)
            arr = _auto_contrast(np.asarray(img, dtype=np.float64), contrast)
            self._base = Image.fromarray(
                np.clip(arr, 0, 255).astype(np.uint8))
            self._key = key
        return self._base

    def sample(self, path: str, size: int = 24, mode: str = "balanced",
               sharpen: float = 0.8, contrast: float = 1.0,
               palette_rgb: np.ndarray | None = None,
               protect_gray: bool = True) -> np.ndarray:
        if mode not in _MODES:
            mode = "balanced"

        if mode == "classic":
            img = Image.open(path).convert("RGB")
            return np.asarray(img.resize((size, size), Image.LANCZOS),
                              dtype=np.uint8)

        if mode == "dither" and palette_rgb is None:
            raise ValueError("dither 模式需要色板 palette_rgb")

        img = self._preprocessed(path, contrast)
        img = _sharpen(img, amount=sharpen)

        if mode == "kmeans":
            img = _kmeans_quantize(img, k=16)

        if mode == "dither":
            base = np.asarray(img.resize((size, size), Image.LANCZOS),
                              dtype=np.float64)
            out = _dither_to_palette(base, palette_rgb, protect_gray)
        else:
            out = _majority_sampling(img, size)
            if mode == "outline":
                mask = _outline_mask(img, size)
                out[mask] = (20, 20, 20)  # 深灰，匹配时归入色板黑色系
        return np.clip(out, 0, 255).astype(np.uint8)
