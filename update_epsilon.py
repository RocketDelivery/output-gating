import tensorflow as tf
from output_gate import Conv2DOutputGate


class UpdateEpsilon(tf.keras.callbacks.Callback):
    def __init__(self, start_epsilon, end_epsilon, end_epoch):
        super().__init__()
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.end_epoch = end_epoch

    def on_epoch_begin(self, epoch, logs):
        if epoch <= self.end_epoch:
            cur_epsilon = (self.end_epsilon - self.start_epsilon) / self.end_epoch * epoch + self.start_epsilon
        else:
            cur_epsilon = self.end_epsilon
        print(f'Current epsilon: {cur_epsilon}')
        
        for layer in self.model.layers: # type: ignore
            if type(layer) is Conv2DOutputGate:
                layer.epsilon.assign([cur_epsilon])
