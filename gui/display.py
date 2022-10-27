from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QGridLayout, QLabel, QSpinBox, QSlider, QCheckBox
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot


class DisplaySettingsWidget(QWidget):
    opacity_changed = pyqtSignal(float)
    input_idx_changed = pyqtSignal(int)

    def __init__(self, num_inputs):
        super().__init__()
        root_layout = QVBoxLayout()
        self.setLayout(root_layout)

        disp_groupbox = QGroupBox('Display')
        disp_layout = QGridLayout()
        disp_groupbox.setLayout(disp_layout)
        root_layout.addWidget(disp_groupbox)

        disp_layout.addWidget(QLabel('Input image:'), 0, 0)
        self._img_spinbox = QSpinBox()
        self._img_spinbox.setRange(1, num_inputs)
        disp_layout.addWidget(self._img_spinbox, 0, 1)
        self._img_max_lbl = QLabel(f'(of {num_inputs})')
        disp_layout.addWidget(self._img_max_lbl, 0, 2)

        disp_layout.addWidget(QLabel('Ofmap channel:'), 1, 0)
        self.ch_idx_spinbox = QSpinBox()
        disp_layout.addWidget(self.ch_idx_spinbox, 1, 1)
        self.ch_idx_max_lbl = QLabel('(Max: N/A)')
        disp_layout.addWidget(self.ch_idx_max_lbl, 1, 2)

        disp_layout.addWidget(QLabel('Overlay image opacity'), 2, 0, 1, 3)
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setRange(0, 255)
        opacity_slider.setTickInterval(1)
        opacity_slider.setValue(127)
        disp_layout.addWidget(opacity_slider, 3, 0, 1, 3)
        
        self._img_spinbox.valueChanged.connect(self._on_image_idx_changed)
        opacity_slider.valueChanged.connect(self._on_opacity_changed)
        
    def histogram_pruning_only(self):
        pass

    @pyqtSlot(int)
    def _on_image_idx_changed(self, value):
        self.input_idx_changed.emit(value-1)

    @pyqtSlot(int)
    def _on_opacity_changed(self, value):
        self.opacity_changed.emit(value / 255)
