"""鼠标输入封装：点击与滚动。

注意：滚轮事件不经过 pyautogui.scroll —— 它把刻度数直接当作 dwData 发送，
而 Windows 应用需要 delta 累积到 ±120（WHEEL_DELTA）才滚动一格，
小 delta 事件会被吞掉（表现为"滚动距离微乎其微"）。
这里用 ctypes 直接发标准 120 倍数的滚轮事件。
FAILSAFE 开启：鼠标猛推到屏幕角落可紧急中断。
"""
from __future__ import annotations

import ctypes
import time

import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

_MOUSEEVENTF_WHEEL = 0x0800
_WHEEL_DELTA = 120  # Windows 标准：一个滚轮刻度 = 120


def click(x: float, y: float, hold_ms: int = 40) -> None:
    """移动到 (x, y) 后 按下 → 保持 → 抬起。

    游戏引擎通常需要鼠标按下状态持续至少一帧（~16ms）才注册点击，
    瞬发 click 容易被吞，所以默认按住 40ms。
    """
    pyautogui.moveTo(int(x), int(y))
    pyautogui.mouseDown()
    time.sleep(max(hold_ms, 1) / 1000.0)
    pyautogui.mouseUp()


def sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def _wheel(notches: int) -> None:
    """发送标准滚轮事件。notches 为刻度数，正=向上，负=向下。"""
    ctypes.windll.user32.mouse_event(
        _MOUSEEVENTF_WHEEL, 0, 0, int(notches) * _WHEEL_DELTA, 0)


def _scroll(notches: int, over_x: int, over_y: int,
            settle_ms: int, hover_ms: int, pulse_ms: int) -> None:
    """移到指定位置 → hover 停顿 → 逐刻度连发滚轮事件 → 等待稳定。

    逐刻度连发（而不是一次发大 delta）：有些游戏 UI 对滚轮做平滑滚动
    动画，单次大 delta 会被限幅吞掉，连发才能滚足距离。
    hover 停顿：防止滚轮事件落在画布上触发画布缩放。
    """
    pyautogui.moveTo(int(over_x), int(over_y))
    sleep_ms(hover_ms)
    step = 1 if notches > 0 else -1
    for _ in range(abs(notches)):
        _wheel(step)
        sleep_ms(pulse_ms)
    sleep_ms(settle_ms)


def scroll_down(notches: int, over_x: int, over_y: int, settle_ms: int,
                hover_ms: int = 60, pulse_ms: int = 50) -> None:
    """向下滚动 notches 个刻度（逐刻度连发）。"""
    _scroll(-abs(notches), over_x, over_y, settle_ms, hover_ms, pulse_ms)


def scroll_up(notches: int, over_x: int, over_y: int, settle_ms: int,
              hover_ms: int = 60, pulse_ms: int = 50) -> None:
    """向上滚动 notches 个刻度（逐刻度连发）。"""
    _scroll(abs(notches), over_x, over_y, settle_ms, hover_ms, pulse_ms)
