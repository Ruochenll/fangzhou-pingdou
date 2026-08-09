"""OpenCV 通用工具：色块轮廓检测、行分组、画布（白色大矩形）检测。"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_background(bgr: np.ndarray, quant: int = 8) -> np.ndarray:
    """估计背景色：量化后取众数。UI 面板的纯色背景在众数上占绝对优势。"""
    q = (bgr // quant * quant).reshape(-1, 3)
    colors, counts = np.unique(q, axis=0, return_counts=True)
    return colors[np.argmax(counts)].astype(np.float64) + quant / 2


def find_swatch_boxes(bgr: np.ndarray, min_side: int = 14, max_side: int = 300,
                      bg_tol: int | None = None, top_colors: int = 80):
    """在图像中检测实心方形色块（调色板色块）。

    PNG 截图中色块内部是像素级纯色，因此按"精确颜色"提取连通域最稳：
    对出现次数最多的若干种颜色分别做 connectedComponents，
    再按边长范围、长宽比、实心度过滤，取中心小块均值作为色块颜色。

    bg_tol: 与估计背景色的逐通道容差，<= 此值的颜色被当作背景跳过；
            传 None 则不做背景排除（由调用方按位置自行过滤）。
    返回 [(cx, cy, w, h, (r, g, b))]，未排序。坐标相对输入图像。
    """
    bg = estimate_background(bgr) if bg_tol is not None else None
    flat = bgr.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)

    min_area = int(min_side * min_side * 0.7)
    boxes = []
    for ci in order[:top_colors]:
        if counts[ci] < min_area:
            break
        color = colors[ci]
        if bg is not None and np.abs(color.astype(np.float64) - bg).max() <= bg_tol:
            continue  # 背景色系
        mask = np.all(bgr == color, axis=2).astype(np.uint8)
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (min_side <= w <= max_side and min_side <= h <= max_side):
                continue
            if not (0.75 <= w / max(h, 1) <= 1.33):
                continue
            if area < min_area or area / float(w * h) < 0.8:  # 实心度
                continue
            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            r = max(2, min(w, h) // 6)  # 中心采样半径，避开选中描边
            patch = bgr[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
            mean_bgr = patch.reshape(-1, 3).mean(axis=0)
            rgb = (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))
            boxes.append((cx, cy, w, h, rgb))
    return boxes


def find_dark_panels(bgr: np.ndarray, min_area_ratio: float = 0.03,
                     dark_lo: int = 25, dark_hi: int = 115):
    """找大而暗的矩形区域（色板、导航器等深色面板）。

    不依赖背景众数，直接按亮度范围找暗色连通域；色块间隙的暗色网络
    保证面板是一个连通整体。返回 [(x, y, w, h, area)] 按面积降序。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = ((gray >= dark_lo) & (gray <= dark_hi)).astype(np.uint8)
    # 闭运算填掉面板上文字/图标造成的小亮洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _labels, stats, _cents = cv2.connectedComponentsWithStats(mask, 8)
    h_img, w_img = gray.shape
    panels = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < w_img * h_img * min_area_ratio:
            continue
        if w < 60 or h < 80:  # 面板至少有一定体量
            continue
        panels.append((x, y, w, h, area))
    panels.sort(key=lambda p: -p[4])
    return panels


def group_rows(boxes, tol_ratio: float = 0.5):
    """把色块按中心 y 聚成行，行内按 x 排序。返回 [[box, ...], ...]（按行排序）。"""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[1])
    rows: list[list] = []
    for b in boxes:
        for row in rows:
            if abs(row[0][1] - b[1]) <= row[0][2] * tol_ratio:
                row.append(b)
                break
        else:
            rows.append([b])
    for row in rows:
        row.sort(key=lambda b: b[0])
    rows.sort(key=lambda row: sum(b[1] for b in row) / len(row))
    return rows


def find_bright_canvas(bgr: np.ndarray, thresh: int = 235):
    """找图中最大的亮白色矩形区域（24×24 画布）。返回 (x, y, w, h) 或 None。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    # 闭运算把网格灰线造成的空洞填掉
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h_img, w_img = gray.shape
    best, best_area = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area <= best_area:
            continue
        if not (0.6 <= w / max(h, 1) <= 1.6):  # 画布接近正方形
            continue
        if area < w_img * h_img * 0.05:  # 至少占 ROI 5%
            continue
        best, best_area = (x, y, w, h), area
    return best
