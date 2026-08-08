"""全局配置：区域坐标、匹配阈值、自动化参数。自动持久化到 config.json。"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field


def _is_frozen() -> bool:
    """是否运行在 PyInstaller 打包后的环境里。"""
    return getattr(sys, "frozen", False)


def resource_path(name: str) -> str:
    """只读资源的绝对路径（known_palette.json 等）。

    打包后这些资源被解压到 PyInstaller 的 _MEIPASS 临时目录（只读）；
    开发时直接从源码目录读取。
    """
    if _is_frozen():
        return os.path.join(sys._MEIPASS, name)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


# 可写文件目录：开发时=源码目录，打包后=exe 所在目录（exe 同级，用户可见可改）
DATA_DIR = (os.path.dirname(sys.executable) if _is_frozen()
            else os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
DEBUG_DIR = os.path.join(DATA_DIR, "debug")
PALETTE_PATH = resource_path("known_palette.json")

GRID_SIZE = 24


@dataclass
class Region:
    """矩形区域。ROI 用屏幕绝对坐标；grid / palette 用相对 ROI 坐标。"""

    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def is_valid(self) -> bool:
        return self.w > 10 and self.h > 10

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @classmethod
    def from_tuple(cls, t) -> "Region":
        return cls(int(t[0]), int(t[1]), int(t[2]), int(t[3]))


@dataclass
class AppConfig:
    roi: Region = field(default_factory=Region)
    match_threshold: float = 12.0      # CIEDE2000：图片像素 → 色板 的匹配阈值
    swatch_threshold: float = 10.0     # CIEDE2000：屏幕色块 → 已知色板 的识别阈值
    scroll_step: int = 15              # 每轮向下滚动的滚轮刻度数（实测约 40 刻度滚完整个色板）
    scroll_pulse_ms: int = 50          # 逐刻度连发滚轮事件的间隔
    scroll_settle_ms: int = 400        # 滚动后等待界面稳定
    click_interval_ms: int = 60        # 连续点击格子的间隔
    swatch_click_settle_ms: int = 120  # 点击色板后的等待
    start_delay_s: int = 3             # 开始填充前的倒计时

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "AppConfig":
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return cfg
        roi = data.pop("roi", None)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        if roi:
            cfg.roi = Region(**roi)
        return cfg
