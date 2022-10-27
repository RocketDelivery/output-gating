import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPaintEvent, QPainter, QColor, QPen, QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, QRect


class GridWidget(QWidget):
    def __init__(self, col_width, row_height, border_thickness, overlay_image):
        super().__init__()
        self._col_width = col_width
        self._row_height = row_height
        self._border_thickness = border_thickness 
        self._color_grid = None
        self._block_dim = None
        self._image = overlay_image if overlay_image is not None else None
        if self._image is not None:
            self._image = self._add_alpha_channel(self._image)
        self._image_opacity = 0.5
        self._update_image(self._image, self._image_opacity)
        
    def paintEvent(self, e: QPaintEvent):
        painter = QPainter(self)
        if self._color_grid is not None:
            grid_rows = self._color_grid.shape[1]
            grid_cols = self._color_grid.shape[2]

            row_start = np.floor(e.rect().top() / self._row_height).astype(np.int32)
            row_end = np.ceil(e.rect().bottom() / self._row_height).astype(np.int32)
            col_start = np.floor(e.rect().left() / self._col_width).astype(np.int32)
            col_end = np.ceil(e.rect().right() / self._col_width).astype(np.int32)

            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    painter.fillRect(
                        QRect(
                            int(col * self._col_width), 
                            int(row * self._row_height), 
                            int(self._col_width), 
                            int(self._row_height)
                        ),
                        QColor(*self._color_grid[0, row, col])
                    )
            white_pen = QPen(QColor(255, 255, 255), self._border_thickness)
            painter.setPen(white_pen)
            if self._block_dim is not None:
                for row in range(0, grid_rows+1, self._block_dim[0]):
                    y_pos = row * self._row_height
                    painter.drawLine(0, y_pos, self.size().width(), y_pos)
                for col in range(0, grid_cols+1, self._block_dim[1]):
                    x_pos = col * self._col_width
                    painter.drawLine(x_pos, 0, x_pos, self.size().height())

            if self._image is not None:
                qimage = QImage(self._image, self._image.shape[1], self._image.shape[0], QImage.Format.Format_RGBA8888) # type: ignore
                qpixmap = QPixmap(qimage)
                painter.drawPixmap(0, 0, self._col_width * grid_cols, self._row_height * grid_rows, qpixmap, 0, 0, 0, 0)

    def update_grid(self, grid, sparse_mask, block_dim):
        self._block_dim = block_dim
        if grid is not None:
            grid_rows = grid.shape[1]
            grid_cols = grid.shape[2]
            intensity_grid = np.clip(np.log10(np.abs(grid) + 1), 0, 1)
            intensity_grid = (200 * intensity_grid).astype(np.uint8)
            self._color_grid = np.zeros(shape=(*grid.shape, 3), dtype=np.uint8)

            pos_mask = grid > 0
            pos_intensity = intensity_grid[pos_mask]
            self._color_grid[pos_mask, :] = np.stack(
                [
                    np.full_like(pos_intensity, 255, dtype=np.uint8), 
                    np.full_like(pos_intensity, 200, dtype=np.uint8) - pos_intensity,
                    np.full_like(pos_intensity, 200, dtype=np.uint8) - pos_intensity
                ], 
                axis=-1)

            neg_mask = grid < 0
            neg_intensity = intensity_grid[neg_mask]
            self._color_grid[neg_mask, :] = np.stack(
                [
                    np.full_like(neg_intensity, 200, dtype=np.uint8) - neg_intensity,
                    np.full_like(neg_intensity, 200, dtype=np.uint8) - neg_intensity,
                    np.full_like(neg_intensity, 255, dtype=np.uint8),
                ],
                axis=-1
            )
            self.setFixedSize(self._col_width * grid_cols, self._row_height * grid_rows)

            zero_mask = grid == 0
            self._color_grid[zero_mask, :] = np.array([240, 240, 240], dtype=np.uint8)
            if sparse_mask is not None:
                self._color_grid[sparse_mask, :] = np.array([10, 10, 10], dtype=np.uint8)
        else:
            self._color_grid = None

        self.update()

    def _update_image(self, image, opacity):
        self._image_opacity = opacity
        if image is not None:
            self._image = image
            self._image[..., 3] = int(opacity * 255)
        else:
            self._image = None

    def _add_alpha_channel(self, image):
        return np.concatenate(
            [
                image, 
                np.full(shape=(*image.shape[0:2], 1), fill_value=255, dtype=np.uint8)
            ], 
            axis=-1
        )

    @pyqtSlot(np.ndarray)
    def on_set_image(self, image):
        self._update_image(self._add_alpha_channel(image), self._image_opacity)
        self.update()

    @pyqtSlot(float)
    def on_set_opacity(self, opacity):
        self._update_image(self._image, opacity)
        self.update()
