"""滚动标定工具：量化游戏对滚轮事件的实际响应。

先回到色板顶部，然后分别以 1 / 3 / 8 / 15 刻度向下滚动，
每次滚动后扫描色板，打印新出现的颜色数——据此判断每刻度滚多少、
scroll_step 设多少合适。

用法（管理员终端）：
    .venv/Scripts/python.exe tools/scroll_test.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig
from core.palette_data import KnownPalette
from core.palette_scanner import PaletteScanner
from utils.input import scroll_down, scroll_up
from utils.screenshot import ScreenCapturer

TOP_MARKER = 0  # 色板第一行黑色


def scan(cap, scanner):
    return set(scanner.scan(cap.grab_region(scanner.roi)).keys())


def main() -> None:
    cfg = AppConfig.load()
    palette = KnownPalette.load()
    if palette is None:
        print("未找到 known_palette.json")
        return
    if not cfg.roi.is_valid():
        print("请先在主程序里框选游戏区域")
        return

    cap = ScreenCapturer()
    scanner = PaletteScanner(palette, cfg.roi, cfg.swatch_threshold)
    sx = cfg.roi.x + int(cfg.roi.w * 0.8)
    sy = cfg.roi.y + int(cfg.roi.h * 0.6)

    input("请把游戏窗口置于前台、色板尽量滚到顶部，然后回车（3 秒后开始）…")
    time.sleep(3)

    # 先确保回到顶部
    for _ in range(40):
        if TOP_MARKER in scan(cap, scanner):
            break
        scroll_up(15, sx, sy, cfg.scroll_settle_ms, 60, cfg.scroll_pulse_ms)
    print("已到色板顶部\n")

    for step in (1, 3, 8, 15):
        before = scan(cap, scanner)
        scroll_down(step, sx, sy, cfg.scroll_settle_ms, 60, cfg.scroll_pulse_ms)
        after = scan(cap, scanner)
        new = after - before
        gone = before - after
        print(f"滚动 {step:2d} 刻度 → 新出现 {len(new)} 色 {sorted(new)}，"
              f"移出 {len(gone)} 色，当前可见 {len(after)} 色")

    print("\n参考：色板共 40 色 / 10 排；若滚 8 刻度能稳定带来约 4 个新色（1 排），")
    print("说明响应正常；若远少于此，说明游戏对滚轮事件限幅，需调大 scroll_step。")

    # 滚回顶部收尾
    for _ in range(40):
        if TOP_MARKER in scan(cap, scanner):
            break
        scroll_up(15, sx, sy, cfg.scroll_settle_ms, 60, cfg.scroll_pulse_ms)
    print("已滚回顶部")


if __name__ == "__main__":
    main()
