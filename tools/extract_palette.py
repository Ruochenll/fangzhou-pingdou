"""从游戏色板截图提取完整色板，生成 known_palette.json。

用法：
    python tools/extract_palette.py <顶部截图> <滚动后截图> [更多截图...] [-o known_palette.json]

截图按"色板从上到下"的顺序传入；相邻截图允许有重叠行，会自动按颜色去重并保持顺序。
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.opencv_helper import find_swatch_boxes, group_rows

COLUMNS = 4


def imread_unicode(path: str) -> np.ndarray:
    """cv2.imread 不支持非 ASCII 路径，用 imdecode 兜底。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片：{path}")
    return img


def name_color(rgb) -> str:
    """根据 RGB 给颜色起个语义化中文名（粗略，仅用于展示）。"""
    r, g, b = [v / 255.0 for v in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360.0
    if s < 0.12:
        for lim, nm in ((0.15, "黑色"), (0.35, "深灰"), (0.60, "灰色"),
                        (0.85, "浅灰"), (1.01, "白色")):
            if v < lim:
                return nm
    base = "红色"
    for lim, nm in ((15, "红色"), (45, "橙色"), (75, "黄色"), (105, "黄绿色"),
                    (165, "绿色"), (200, "青色"), (255, "蓝色"), (290, "紫色"),
                    (330, "品红色"), (361, "红色")):
        if h < lim:
            base = nm
            break
    if base == "红色" and v > 0.8 and s < 0.55:
        base = "粉色"
    if base in ("橙色", "黄色") and v < 0.6 and s > 0.5:
        base = "棕色"
    if v < 0.45:
        return "深" + base
    if v > 0.85 and s < 0.45:
        return "浅" + base
    return base


def extract_rows(path: str, min_side: int = 40) -> list[list[tuple[int, int, int]]]:
    """从一张截图提取色块行，返回 [[(r,g,b)×4], ...]（按从上到下）。"""
    img = imread_unicode(path)
    boxes = find_swatch_boxes(img, min_side=min_side, max_side=220, bg_tol=10)
    rows = []
    for row in group_rows(boxes):
        if len(row) == COLUMNS:
            rows.append([b[4] for b in row])
    return rows


def rows_match(r1, r2, tol: float = 24.0) -> bool:
    """两行颜色是否相同（逐列 RGB 欧氏距离均值）。"""
    if len(r1) != len(r2):
        return False
    d = np.mean([np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float))
                 for a, b in zip(r1, r2)])
    return d < tol


def main() -> None:
    ap = argparse.ArgumentParser(description="从色板截图提取完整色板")
    ap.add_argument("shots", nargs="+", help="色板截图，按从上到下顺序")
    ap.add_argument("-o", "--output",
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), "known_palette.json"))
    args = ap.parse_args()

    merged: list[list] = []
    for shot in args.shots:
        rows = extract_rows(shot)
        print(f"{os.path.basename(shot)}: 识别到 {len(rows)} 行完整色块")
        for row in rows:
            if not any(rows_match(row, m) for m in merged):
                merged.append(row)

    colors = []
    name_count: dict[str, int] = {}
    idx = 0
    for ri, row in enumerate(merged):
        for ci, rgb in enumerate(row):
            name = name_color(rgb)
            name_count[name] = name_count.get(name, 0) + 1
            if name_count[name] > 1:
                name = f"{name}{name_count[name]}"
            colors.append({"index": idx, "row": ri, "col": ci,
                           "rgb": [int(v) for v in rgb], "name": name})
            idx += 1

    out = {"grid_columns": COLUMNS, "colors": colors}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n共提取 {idx} 色 → {args.output}\n")
    for c in colors:
        print(f"  [{c['index']:2d}] 行{c['row']}列{c['col']}  "
              f"rgb{tuple(c['rgb'])}  {c['name']}")


if __name__ == "__main__":
    main()
