"""24×24 像素画预览：显示匹配到色板后的颜色，超阈值格子红框警示，已填格子打勾。"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class PixelCanvasView(QWidget):
    def __init__(self, size: int = 24, parent=None) -> None:
        super().__init__(parent)
        self.size = size
        self.setMinimumSize(380, 380)
        self._colors: np.ndarray | None = None   # (24,24,3) 匹配后的色板 RGB
        self._warn: np.ndarray | None = None     # (24,24) bool 超出匹配阈值
        self._painted: np.ndarray | None = None  # (24,24) bool 已填充

    def set_data(self, colors: np.ndarray, warn: np.ndarray) -> None:
        self._colors = colors
        self._warn = warn
        self._painted = np.zeros((self.size, self.size), dtype=bool)
        self.update()

    def mark_painted(self, r: int, c: int) -> None:
        if self._painted is not None:
            self._painted[r, c] = True
            self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(245, 245, 245))
        if self._colors is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter, "导入图片后在此预览 24×24 像素画")
            return

        n = self.size
        cw = self.width() / n
        ch = self.height() / n
        for r in range(n):
            for c in range(n):
                rgb = self._colors[r, c]
                p.fillRect(QRectF(c * cw, r * ch, cw + 0.5, ch + 0.5),
                           QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])))
                if self._warn is not None and self._warn[r, c]:
                    p.setPen(QPen(QColor(220, 60, 60), 2))
                    p.drawRect(QRectF(c * cw + 1, r * ch + 1, cw - 2, ch - 2))
                if self._painted is not None and self._painted[r, c]:
                    p.fillRect(QRectF(c * cw, r * ch, cw + 0.5, ch + 0.5),
                               QColor(255, 255, 255, 90))

        p.setPen(QPen(QColor(0, 0, 0, 36), 1))
        for i in range(n + 1):
            p.drawLine(QPointF(0, i * ch), QPointF(self.width(), i * ch))
            p.drawLine(QPointF(i * cw, 0), QPointF(i * cw, self.height()))
