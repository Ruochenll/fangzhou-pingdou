"""拼豆像素画助手 · 入口。

把任意图片转换为 24×24 像素画，匹配游戏固定色板，
再通过图像识别 + 模拟点击在游戏涂鸦玩法中自动落格。
"""
import ctypes
import os
import sys

# Windows 显示缩放非 100% 时，保证 mss 截图与 pyautogui 点击使用同一套物理像素坐标
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_DPI_AWARE
    except Exception:
        pass

# 关闭 Qt 高 DPI 缩放：让框选 ROI 的坐标 = mss 截图坐标 = pyautogui 点击坐标（全部物理像素）
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("拼豆像素画助手")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
