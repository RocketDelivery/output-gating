class DefaultVisualizer():
    def num_output_channels(self, layer, output):
        return output.numpy().shape[3]

    def weight_grid(self, layer, output, output_ch_idx):
        return None, None

    def histogram_values(self, layer, output):
        return output.numpy(), None

    def output_grid(self, layer, output, output_ch_idx):
        return output.numpy()[:, :, :, output_ch_idx], None

    def channel_grid(self, layer, output, row_sel, col_sel):
        return output.numpy()[:, row_sel, col_sel, :], None


class Conv2DVisualizer(DefaultVisualizer):
    def weight_grid(self, layer, output, output_ch_idx):
        weights = layer.weights[0].numpy()
        weights_sel = weights[:, :, :, output_ch_idx]
        weights_sel = weights_sel.reshape((-1, weights_sel.shape[2])).T
        return weights_sel, None


class DepthwiseConv2DVisualizer(DefaultVisualizer):
    def weight_grid(self, layer, output, output_ch_idx):
        weights = layer.weights[0].numpy()
        weights_sel = weights[:, :, output_ch_idx, 0]
        weights_sel = weights_sel.reshape((-1, 1)).T
        return weights_sel, None


class Conv2DOutputGateVisualizer(Conv2DVisualizer):
    def num_output_channels(self, layer, output):
        return super().num_output_channels(layer, output[0])

    def weight_grid(self, layer, output, output_ch_idx):
        weights = layer.weights[1].numpy()
        weights_sel = weights[:, :, :, output_ch_idx]
        weights_sel = weights_sel.reshape((-1, weights_sel.shape[2])).T
        return weights_sel, None

    def histogram_values(self, layer, output):
        if layer.intermediate_output is not None:
            return output[0].numpy(), layer.intermediate_output == 0
        else:
            return output[0].numpy(), None

    def output_grid(self, layer, output, output_ch_idx):
        if layer.intermediate_output is not None:
            return output[0].numpy()[:, :, :, output_ch_idx], layer.intermediate_output[:, :, :, output_ch_idx] == 0
        else:
            return output[0].numpy()[:, :, :, output_ch_idx], None 

    def channel_grid(self, layer, output, row_sel, col_sel):
        if layer.intermediate_output is not None:
            return output[0].numpy()[:, row_sel, col_sel, :], layer.intermediate_output[:, row_sel, col_sel, :] == 0
        else:
            return output[0].numpy()[:, row_sel, col_sel, :], None
