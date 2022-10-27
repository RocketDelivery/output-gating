import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import numpy as np


class HistogramWidget(FigureCanvasQTAgg):
    def __init__(self):
        fig = plt.Figure(figsize=(1, 1)) # type: ignore
        self.ax_hist = fig.add_subplot(111)
        super(HistogramWidget, self).__init__(fig)

    def update_values(self, values, output_nulls):
        values = np.abs(values).reshape(-1)
        if output_nulls is None:
            nulls = np.zeros_like(values, dtype=np.bool)
        else:
            nulls = output_nulls.reshape(-1)
        num_zeros = np.count_nonzero(np.logical_and(values == 0, ~nulls))
        num_nulls = np.count_nonzero(nulls)
        # TODO: programatically create the bins
        bin = np.zeros(18)
        bin[0] = num_nulls
        bin[1] = num_zeros
        bin[2] = np.count_nonzero(np.logical_and(0.1 >= values, values > 0))
        bin[3] = np.count_nonzero(np.logical_and(0.2 >= values, values > 0.1))
        bin[4] = np.count_nonzero(np.logical_and(0.3 >= values, values > 0.2))
        bin[5] = np.count_nonzero(np.logical_and(0.4 >= values, values > 0.3))
        bin[6] = np.count_nonzero(np.logical_and(0.5 >= values, values > 0.4))
        bin[7] = np.count_nonzero(np.logical_and(0.6 >= values, values > 0.5))
        bin[8] = np.count_nonzero(np.logical_and(0.7 >= values, values > 0.6))
        bin[9] = np.count_nonzero(np.logical_and(0.8 >= values, values > 0.7))
        bin[10] = np.count_nonzero(np.logical_and(0.9 >= values, values > 0.8))
        bin[11] = np.count_nonzero(np.logical_and(1.0 >= values, values > 0.9))
        bin[12] = np.count_nonzero(np.logical_and(1.1 >= values, values > 1.0))
        bin[13] = np.count_nonzero(np.logical_and(1.2 >= values, values > 1.1))
        bin[14] = np.count_nonzero(np.logical_and(1.3 >= values, values > 1.2))
        bin[15] = np.count_nonzero(np.logical_and(1.4 >= values, values > 1.3))
        bin[16] = np.count_nonzero(np.logical_and(1.5 >= values, values > 1.4))
        bin[17] = np.count_nonzero(values > 1.5)
        ticks = [
            "null",
            "0",
            "(0, 0.1]",
            "(0.1, 0.2]",
            "(0.2, 0.3]",
            "(0.3, 0.4]",
            "(0.4, 0.5]",
            "(0.5, 0.6]",
            "(0.6, 0.7]",
            "(0.7, 0.8]",
            "(0.8, 0.9]",
            "(0.9, 1.0]",
            "(1.0, 1.1]",
            "(1.1, 1.2]",
            "(1.2, 1.3]",
            "(1.3, 1.4]",
            "(1.4, 1.5]",
            "(1.5, inf)",
        ]
        self.ax_hist.clear()
        self.ax_hist.bar(np.arange(bin.size), bin, width=0.95)
        self.ax_hist.set_xticks(np.arange(bin.size), ticks, rotation=45)
        self.ax_hist.set_title(
            f'Output Data Sparsity, {num_nulls}/{values.size} nulls ({num_nulls/values.size*100:.1f}% output sparsity)')
        self.draw()
