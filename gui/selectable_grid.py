from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QRect

from gui.grid import GridWidget


class SelectableGridWidget(GridWidget):
    grid_selected = pyqtSignal(int, int)
    grid_cleared = pyqtSignal()

    def __init__(self, col_width, row_height, border_thickness, overlay_image):
        super().__init__(col_width, row_height, border_thickness, overlay_image)
        self._sel_pos = None
        self.setMouseTracking(True)

    def update_grid(self, grid, sparse_mask, block_dim):
        super().update_grid(grid, sparse_mask, block_dim)
        self._sel_pos = None

    def mousePressEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            x_pos = int(e.x() / self._col_width)
            y_pos = int(e.y() / self._row_height)
            self._sel_pos = (x_pos, y_pos)
            self.grid_selected.emit(y_pos, x_pos)
            self.update()
        elif e.buttons() & Qt.MouseButton.RightButton:
            self._sel_pos = None
            self.grid_cleared.emit()
            self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._sel_pos is not None:
            painter = QPainter(self)
            border_pen = QPen(QColor(32, 144, 140), 3)
            painter.setPen(border_pen)
            painter.drawRect(QRect(
                self._sel_pos[0] * self._col_width,
                self._sel_pos[1] * self._row_height,
                self._col_width,
                self._row_height
            ))
