"""落格工作线程：扫描可见色板 → 按色聚类点击 → 滚动 → 循环直到完成或到底。

滚动策略（自适应）：
- 开始前先向上滚回色板顶部（以第一行黑色色块出现为判据），避免上次停在中途
- 每轮滚动后重新扫描；出现新颜色就继续填，连续 2 轮滚动无新色才判定到底
- 该游戏滚轮步进很小，宁多滚勿漏滚

安全机制：
- pyautogui FAILSAFE：鼠标猛推到屏幕角落立即中断
- stop()：由 GUI 停止按钮或 ESC 热键触发
"""
from __future__ import annotations

import time

from PySide6.QtCore import QThread, Signal

from core.grid_locator import locate_grid
from core.palette_scanner import PaletteScanner
from utils.input import click, scroll_down, scroll_up, sleep_ms
from utils.screenshot import ScreenCapturer

# 色板第一行第一个颜色的索引（已知色板行0列0=黑色），用于回到顶部判定
_TOP_MARKER_INDEX = 0
# 回滚到顶部最多尝试次数
_MAX_SCROLL_UP = 40
# 连续多少轮滚动无新色判定为到底（兜底判据）
_STALE_LIMIT = 5
# 无新色时每轮额外多滚的刻度数（滚动量递增补偿）
_STALE_BONUS = 4
# 主循环最大轮数（防死循环）
_MAX_ROUNDS = 30


class PaintWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)              # 已填, 总数
    cell_done = Signal(int, int, int)        # row, col, palette_idx
    finished_all = Signal()                  # 全部填完
    finished_partial = Signal(int)           # 提前结束，参数 = 剩余未填格数
    failed = Signal(str)

    def __init__(self, plan: dict, palette, roi, cfg, parent=None) -> None:
        """
        plan: {palette_idx: [(row, col), ...]} 按色聚类的落格计划。
        palette: KnownPalette；roi: Region（屏幕绝对坐标）；cfg: AppConfig。
        """
        super().__init__(parent)
        self.plan = plan
        self.palette = palette
        self.roi = roi
        self.cfg = cfg
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:  # 含 pyautogui.FailSafeException
            self.failed.emit(str(e))

    # ---------- 内部 ----------

    def _scroll_point(self) -> tuple[int, int]:
        """滚动时鼠标的悬停位置（色板区域上方）。"""
        return (self.roi.x + int(self.roi.w * 0.8),
                self.roi.y + int(self.roi.h * 0.6))

    def _back_to_top(self, cap, scanner) -> None:
        """向上滚动直到色板第一行（黑色）出现。"""
        for _ in range(_MAX_SCROLL_UP):
            if self._stop:
                return
            visible = scanner.scan(cap.grab_region(self.roi))
            if _TOP_MARKER_INDEX in visible:
                self.log.emit("色板已回到顶部")
                return
            sx, sy = self._scroll_point()
            scroll_up(self.cfg.scroll_step, sx, sy,
                      self.cfg.scroll_settle_ms, 60, self.cfg.scroll_pulse_ms)
        self.log.emit("⚠ 回滚多次仍未见色板首行，从当前位置继续")

    def _run(self) -> None:
        total = sum(len(v) for v in self.plan.values())
        if total == 0:
            self.failed.emit("落格计划为空（可能所有格子都被跳过了）")
            return

        cap = ScreenCapturer()
        scanner = PaletteScanner(self.palette, self.roi, self.cfg.swatch_threshold)

        self.log.emit(f"落格计划：{total} 格 / {len(self.plan)} 种颜色")
        for i in range(self.cfg.start_delay_s, 0, -1):
            if self._stop:
                break
            self.log.emit(f"{i} 秒后开始…请确保游戏窗口完整可见、未被遮挡")
            time.sleep(1)
        if self._stop:
            self.finished_partial.emit(total)
            return

        # —— 定位 24×24 画布（只做一次，画布不会移动）——
        roi_img = cap.grab_region(self.roi)
        grid = locate_grid(roi_img, self.roi)
        if grid is None:
            self.failed.emit("未能在框选区域内找到 24×24 画布，请调整框选范围后重试")
            return
        self.log.emit(f"画布定位完成：格子 {grid.cell_w:.1f}×{grid.cell_h:.1f} px")

        # —— 先回滚到色板顶部 ——
        self._back_to_top(cap, scanner)

        remaining = {rc for cells in self.plan.values() for rc in cells}
        done = 0
        seen_all: set[int] = set()   # 扫描到过的所有色板索引
        stale = 0                    # 连续"滚动后无新色"轮数
        # 已知色板最后一排的索引：看到它们 = 真的到底了（主判据）
        bottom_row = set(range(len(self.palette) - self.palette.columns,
                               len(self.palette)))

        # —— 主循环：填当前可见颜色 → 滚动 → 判定到底 ——
        for round_no in range(_MAX_ROUNDS):
            if not remaining or self._stop:
                break

            roi_img = cap.grab_region(self.roi)
            visible = scanner.scan(roi_img)
            new_colors = set(visible.keys()) - seen_all
            seen_all |= set(visible.keys())

            for idx in sorted(self.plan.keys()):
                if self._stop or not remaining:
                    break
                if idx not in visible:
                    continue
                todo = [rc for rc in self.plan[idx] if rc in remaining]
                if not todo:
                    continue

                sx, sy = visible[idx]
                click(sx, sy)
                sleep_ms(self.cfg.swatch_click_settle_ms)
                self.log.emit(f"选用「{self.palette.name(idx)}」，填充 {len(todo)} 格")

                for (r, c) in todo:
                    if self._stop:
                        break
                    gx, gy = grid.cell_center(r, c)
                    click(gx, gy)
                    remaining.discard((r, c))
                    done += 1
                    self.cell_done.emit(r, c, idx)
                    self.progress.emit(done, total)
                    sleep_ms(self.cfg.click_interval_ms)

            if not remaining:
                break

            # —— 到底判定（双保险）——
            if bottom_row & set(visible.keys()):
                self.log.emit("已识别到色板最后一排颜色，本屏处理完毕即到底")
                break
            if new_colors:
                stale = 0
            else:
                stale += 1
            if stale >= _STALE_LIMIT:
                self.log.emit("连续多轮滚动无新增颜色，判定已到色板底部")
                break

            # 滚动：无新色时逐轮加量，补偿偶发的滚动量不足
            step = self.cfg.scroll_step + stale * _STALE_BONUS
            sx, sy = self._scroll_point()
            scroll_down(step, sx, sy,
                        self.cfg.scroll_settle_ms, 60, self.cfg.scroll_pulse_ms)
        else:
            self.log.emit(f"⚠ 已达最大轮数 {_MAX_ROUNDS}，强制结束")

        if remaining:
            self.finished_partial.emit(len(remaining))
        else:
            self.finished_all.emit()
