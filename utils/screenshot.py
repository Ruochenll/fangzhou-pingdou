"""基于 mss 的屏幕截图，返回 BGR numpy 数组（供 OpenCV 使用）。"""
from __future__ import annotations

import mss
import numpy as np


class ScreenCapturer:
    """屏幕截图器。mss 返回物理像素坐标，与 pyautogui 的点击坐标一致。"""

    def __init__(self) -> None:
        self._sct = mss.mss()

    def grab(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """截取屏幕指定区域，返回 (h, w, 3) BGR uint8。"""
        raw = self._sct.grab({"left": int(x), "top": int(y),
                              "width": int(w), "height": int(h)})
        return np.asarray(raw)[:, :, :3].copy()  # BGRA -> BGR

    def grab_region(self, region) -> np.ndarray:
        return self.grab(region.x, region.y, region.w, region.h)
