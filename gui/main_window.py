import tensorflow as tf
import numpy as np
from PyQt5.QtWidgets import QWidget, QGridLayout, QScrollArea, QSplitter, QListWidget
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal

from gui.grid import GridWidget
from gui.selectable_grid import SelectableGridWidget
from gui.histogram import HistogramWidget
from gui.display import DisplaySettingsWidget
from output_gate import Conv2DOutputGate
from gui.visualizers import DefaultVisualizer, Conv2DVisualizer, DepthwiseConv2DVisualizer, Conv2DOutputGateVisualizer

visualizer_dict = dict()

def registerVisualizer(layer_type, visualizer):
    visualizer_dict[layer_type] = visualizer

registerVisualizer(tf.keras.layers.Conv2D, Conv2DVisualizer())
registerVisualizer(tf.keras.layers.DepthwiseConv2D, DepthwiseConv2DVisualizer())
registerVisualizer(Conv2DOutputGate, Conv2DOutputGateVisualizer())


class MainWindow(QWidget):
    image_changed = pyqtSignal(np.ndarray)

    def __init__(self, model_path, sample_data, images):
        super().__init__()
        self._model = tf.keras.models.load_model(model_path, custom_objects={'Conv2DOutputGate': Conv2DOutputGate})
        if self._model is None:
            raise RuntimeError('Model is null')
        
        for layer in self._model.layers:
            if type(layer) is Conv2DOutputGate:
                layer.capture_output = True
            
        self._temp_model = self._model
        self._sample_data = sample_data
        self._cur_input_idx = 0
        self._images = images.numpy().astype(np.uint8)
        self._interm_output = None
        self._default_vis = DefaultVisualizer()

        root_layout = QGridLayout()
        self.setLayout(root_layout)

        left_right_splitter = QSplitter(Qt.Horizontal) # type: ignore
        root_layout.addWidget(left_right_splitter, 0, 0, 2, 1)

        self._weight_grid = GridWidget(20, 20, 5, None)
        weight_grid_area = QScrollArea()
        weight_grid_area.setWidget(self._weight_grid)
        weight_grid_area.setWidgetResizable(True)
        left_right_splitter.addWidget(weight_grid_area)

        top_bottom_splitter = QSplitter(Qt.Vertical) # type: ignore
        left_right_splitter.addWidget(top_bottom_splitter)

        self._output_grid = SelectableGridWidget(10, 10, 3, self._get_image())
        output_grid_area = QScrollArea()
        output_grid_area.setWidget(self._output_grid)
        output_grid_area.setWidgetResizable(True)
        top_bottom_splitter.addWidget(output_grid_area)

        self._output_histogram = HistogramWidget()
        top_bottom_splitter.addWidget(self._output_histogram)

        self._output_feature_grid = GridWidget(20, 20, 5, None)
        self._output_feature_grid.setFixedWidth(20)
        output_feature_grid_area = QScrollArea()
        output_feature_grid_area.setWidget(self._output_feature_grid)
        output_feature_grid_area.setWidgetResizable(True)
        output_feature_grid_area.setFixedWidth(20+20)
        root_layout.addWidget(output_feature_grid_area, 0, 1, 2, 1)

        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_panel_layout = QGridLayout()
        right_panel.setLayout(right_panel_layout)
        root_layout.addWidget(right_panel, 0, 2, 2, 1)

        self._layer_list = QListWidget()
        right_panel_layout.addWidget(self._layer_list, 0, 0)
        self._layer_list.itemSelectionChanged.connect(self._on_list_selected)

        self._disp_widget = DisplaySettingsWidget(self._sample_data.shape[0])
        right_panel_layout.addWidget(self._disp_widget, 1, 0)
        self._disp_widget.ch_idx_spinbox.valueChanged.connect(self._on_ch_idx_changed)

        for layer in self._model.layers:
                self._layer_list.addItem(layer.name)

        self.image_changed.connect(self._output_grid.on_set_image) # type: ignore
        self._output_grid.grid_selected.connect(self._on_output_grid_selected) # type: ignore
        self._output_grid.grid_cleared.connect(self._on_output_grid_cleared) # type: ignore
        self._disp_widget.opacity_changed.connect(self._output_grid.on_set_opacity) # type: ignore
        self._disp_widget.input_idx_changed.connect(self._on_input_idx_changed) # type: ignore

    def _get_input_tensor(self):
        return self._sample_data[self._cur_input_idx, tf.newaxis, ...]

    def _get_image(self):
        return self._images[self._cur_input_idx, ...]

    def _update_output(self):
        if self._model is None:
            raise RuntimeError('Model is null')

        sel_index = self._layer_list.selectionModel().selectedIndexes()[0].row()
        self._temp_model = tf.keras.models.Model(self._model.input, self._model.layers[sel_index].output)
        self._cur_output = self._temp_model(self._get_input_tensor()) # type: ignore
        self._cur_layer = self._model.layers[sel_index]
        visualizer = self._get_visualizer(self._cur_layer)

        num_channels = visualizer.num_output_channels(self._cur_layer, self._cur_output)
        hist_values, hist_nulls = visualizer.histogram_values(self._cur_layer, self._cur_output)

        self._disp_widget.ch_idx_spinbox.setRange(0, num_channels-1)
        self._disp_widget.ch_idx_spinbox.setEnabled(True)
        self._disp_widget.ch_idx_max_lbl.setText(f'(Max: {num_channels-1})')
        m_index = self._disp_widget.ch_idx_spinbox.value()

        self._update_weights_view(visualizer, m_index)
        self._update_outputs_view(visualizer, m_index)
        self._output_feature_grid.update_grid(None, None, None)
        self._output_histogram.update_values(hist_values, hist_nulls)

    def _update_weights_view(self, visualizer, ch_idx):
        weights_sel, sparse_mask = visualizer.weight_grid(self._cur_layer, self._cur_output, ch_idx)
        if weights_sel is not None:
            weights_sel = weights_sel[np.newaxis, ...]
            self._weight_grid.update_grid(weights_sel, sparse_mask, (1, 1))
        else:
            self._weight_grid.update_grid(None, None, None)

    def _update_outputs_view(self, visualizer, ch_idx):
        output_sel, sparse_mask = visualizer.output_grid(self._cur_layer, self._cur_output, ch_idx)
        self._output_grid.update_grid(output_sel, sparse_mask, (1, 1))

    def _get_visualizer(self, layer):
        return visualizer_dict.get(type(layer), self._default_vis)

    @pyqtSlot(int)
    def _on_input_idx_changed(self, idx):
        self._cur_input_idx = idx
        self._update_output()
        self.image_changed.emit(self._get_image())

    @pyqtSlot()
    def _on_list_selected(self):
        self._update_output()
        
    @pyqtSlot()
    def _on_ch_idx_changed(self):
        if self._model is None:
            raise RuntimeError('Model is null')
            
        visualizer = self._get_visualizer(self._cur_layer)
        m_index = self._disp_widget.ch_idx_spinbox.value()

        self._update_weights_view(visualizer, m_index)
        self._update_outputs_view(visualizer, m_index)

    @pyqtSlot(int, int)
    def _on_output_grid_selected(self, row, col):
        if self._cur_output is not None:
            visualizer = self._get_visualizer(self._cur_layer)
            output_sel, sparse_mask = visualizer.channel_grid(self._cur_layer, self._cur_output, row, col)
            output_sel = output_sel[:, :, np.newaxis]
            self._output_feature_grid.update_grid(output_sel, sparse_mask, (1, 1))

    @pyqtSlot()
    def _on_output_grid_cleared(self):
        self._output_feature_grid.update_grid(None, None, None)
