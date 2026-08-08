"""全屏半透明覆盖层：拖拽框选游戏区域，ESC 取消。"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget


class RegionSelector(QWidget):
    region_selected = Signal(QRect)  # 屏幕绝对坐标
    cancelled = Signal()

    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        geo = QGuiApplication.primaryScreen().virtualGeometry()
        self.setGeometry(geo)
        self._origin: QPoint | None = None
        self._rect = QRect()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            p = e.globalPosition().toPoint()
            self._origin = p
            self._rect = QRect(p, p)
            self.update()

    def mouseMoveEvent(self, e) -> None:
        if self._origin is not None:
            self._rect = QRect(self._origin, e.globalPosition().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if e.button() == Qt.LeftButton and self._origin is not None:
            rect = self._rect.normalized()
            self._origin = None
            self.close()
            if rect.width() > 20 and rect.height() > 20:
                self.region_selected.emit(rect)
            else:
                self.cancelled.emit()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()
            self.cancelled.emit()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 100))

        p.setPen(QColor(255, 255, 255))
        p.drawText(QPoint(24, 40), "拖拽框选游戏界面（需同时包含画布和右侧色板），ESC 取消")

        if not self._rect.isNull():
            local = self._rect.translated(-self.geometry().x(), -self.geometry().y())
            # 选中区域镂空，露出桌面
            p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.fillRect(local, Qt.transparent)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setPen(QPen(QColor(23, 162, 184), 2))
            p.drawRect(local)
            p.setPen(QColor(255, 255, 255))
            p.drawText(local.bottomLeft() + QPoint(4, 18),
                       f"{self._rect.width()} × {self._rect.height()}")
