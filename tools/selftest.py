"""离线自检（不需要打开游戏）：

1) 校验 known_palette.json 已生成
2) 在完整游戏界面截图上测试：画布定位 + 当前可见色板扫描
3) 测试图片 → 24×24 → 色板匹配管线（用随机测试图）

用法：python tools/selftest.py [完整游戏截图路径]
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEBUG_DIR, GRID_SIZE, PALETTE_PATH, Region
from core.color_matcher import PaletteMatcher
from core.grid_locator import locate_grid
from core.image_processor import load_and_downsample
from core.palette_data import KnownPalette
from core.palette_scanner import PaletteScanner
from tools.extract_palette import imread_unicode

DEFAULT_SHOT = r"C:\Users\w7992\Pictures\Screenshots\屏幕截图 2026-08-08 202348.png"


def test_detect(shot_path: str, palette: KnownPalette) -> bool:
    print(f"\n== 画布定位 + 色板扫描（{os.path.basename(shot_path)}）==")
    img = imread_unicode(shot_path)
    h, w = img.shape[:2]
    roi = Region(0, 0, w, h)  # 整张截图当作 ROI

    grid = locate_grid(img, roi)
    if grid is None:
        print("  ✗ 画布定位失败")
        return False
    print(f"  ✓ 画布：原点 ({grid.origin_x:.0f}, {grid.origin_y:.0f})，"
          f"格子 {grid.cell_w:.1f}×{grid.cell_h:.1f} px")

    visible = PaletteScanner(palette, roi).scan(img)
    print(f"  ✓ 识别到 {len(visible)} 个可见色块：")
    for idx in sorted(visible):
        sx, sy = visible[idx]
        print(f"    [{idx:2d}] {palette.name(idx):6s} @ ({sx}, {sy})")

    # 画叠加图方便人工核对
    dbg = img.copy()
    x0, y0 = int(grid.origin_x), int(grid.origin_y)
    x1 = int(x0 + grid.cell_w * grid.size)
    y1 = int(y0 + grid.cell_h * grid.size)
    cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 200, 0), 2)
    for r in range(0, grid.size, 4):
        for c in range(0, grid.size, 4):
            cx, cy = grid.cell_center(r, c)
            cv2.circle(dbg, (int(cx), int(cy)), 2, (0, 150, 0), -1)
    for idx, (sx, sy) in visible.items():
        cv2.circle(dbg, (sx, sy), 10, (0, 255, 255), 2)
        cv2.putText(dbg, str(idx), (sx - 8, sy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    out = os.path.join(DEBUG_DIR, "selftest_overlay.png")
    cv2.imwrite(out, dbg)
    print(f"  叠加图已保存：{out}")

    ok = len(visible) >= 20 and 10 <= grid.cell_w <= 60
    print(f"  {'✓ 通过' if ok else '✗ 数量/尺寸异常，请检查叠加图'}")
    return ok


def test_pipeline(palette: KnownPalette) -> bool:
    print("\n== 图像处理管线（随机测试图）==")
    rng = np.random.default_rng(7)
    test_img = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    test_path = os.path.join(DEBUG_DIR, "test_input.png")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(test_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))

    pixels = load_and_downsample(test_path, GRID_SIZE)
    matcher = PaletteMatcher(palette.rgb)
    result = matcher.match(pixels)
    needed = matcher.needed_colors(result)
    print(f"  ✓ 24×24 降采样 + 匹配完成，需要 {len(needed)} 种颜色")
    print(f"  用量前 5：{[(palette.name(i), n) for i, n in needed[:5]]}")
    print(f"  最大色差 ΔE2000 = {result.distances.max():.1f}，"
          f"平均 = {result.distances.mean():.1f}")

    mapped = palette.rgb[result.indices].astype(np.uint8)
    preview = cv2.resize(cv2.cvtColor(mapped, cv2.COLOR_RGB2BGR),
                         (480, 480), interpolation=cv2.INTER_NEAREST)
    out = os.path.join(DEBUG_DIR, "test_preview.png")
    cv2.imwrite(out, preview)
    print(f"  匹配后预览图：{out}")
    return len(needed) > 0


def main() -> None:
    shot = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SHOT

    palette = KnownPalette.load()
    if palette is None:
        print(f"✗ 未找到 {PALETTE_PATH}，请先运行 tools/extract_palette.py")
        sys.exit(1)
    print(f"✓ 色板加载：{len(palette)} 色")

    ok1 = test_detect(shot, palette) if os.path.exists(shot) else False
    if not os.path.exists(shot):
        print(f"  （跳过界面检测，截图不存在：{shot}）")
    ok2 = test_pipeline(palette)

    print("\n== 自检结果 ==")
    print(f"  界面检测：{'通过' if ok1 else '未通过/跳过'}")
    print(f"  图像管线：{'通过' if ok2 else '未通过'}")
    sys.exit(0 if ok2 else 1)


if __name__ == "__main__":
    main()
