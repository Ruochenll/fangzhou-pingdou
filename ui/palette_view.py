"""所需色块清单：色块 + 名称 + 用量，网格布局展示。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget


class NeededColorsView(QWidget):
    COLUMNS = 4

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setSpacing(6)
        self._grid.setContentsMargins(4, 4, 4, 4)

    def set_items(self, items: list[tuple[tuple[int, int, int], str, int]]) -> None:
        """items: [(rgb, name, count)]"""
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, (rgb, name, count) in enumerate(items):
            cell = QWidget()
            lay = QVBoxLayout(cell)
            lay.setContentsMargins(2, 2, 2, 2)
            lay.setSpacing(2)

            sw = QLabel()
            sw.setFixedSize(42, 42)
            sw.setStyleSheet(
                f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]});"
                "border: 1px solid #999; border-radius: 4px;")
            lay.addWidget(sw, 0, Qt.AlignCenter)

            lb = QLabel(f"{name}\n×{count}")
            lb.setAlignment(Qt.AlignCenter)
            lb.setStyleSheet("font-size: 11px; color: #444;")
            lay.addWidget(lb)

            self._grid.addWidget(cell, i // self.COLUMNS, i % self.COLUMNS)
