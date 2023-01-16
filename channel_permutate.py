import tensorflow as tf


class ChannelPermutate(tf.keras.layers.Layer):
    def __init__(self, conv2d_output_gate, alpha, channel_rank_mode):
        super().__init__()

        self.conv2d_output_gate_name = conv2d_output_gate.name
        self.filters = conv2d_output_gate.full_conv.filters

        self.moving_avg_sparsity = self.add_weight(
            shape=(conv2d_output_gate.full_conv.filters,),
            initializer=tf.keras.initializers.Constant(0),
            trainable=False,
            name='moving_average_sparsity',
            dtype=tf.float32
        )

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.channel_rank_mode = channel_rank_mode

    def call(self, inputs, training=None):
        if training:
            if self.channel_rank_mode.lower() == 'abs':
                sparsity_scores = tf.reduce_mean(tf.abs(inputs), axis=[0, 1, 2])
            elif self.channel_rank_mode.lower() == 'linear':
                sparsity_scores = tf.reduce_mean(inputs, axis=[0, 1, 2])
            else:
                raise ValueError(f'Invalid channel_rank_mode parameter: {self.channel_rank_mode}')
            self.moving_avg_sparsity.assign(
                self.moving_avg_sparsity * self.alpha + sparsity_scores * (1 - self.alpha)
            )
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'conv2d_output_gate_name': self.conv2d_output_gate_name,
            'filters': self.filters,
            'alpha': self.alpha,
            'channel_rank_mode': self.channel_rank_mode
        })
        return config

    def apply_permutation(self, conv2d_output_gate):
        channel_rank = tf.concat(
            [
                tf.constant([0], dtype=tf.int32),
                tf.argsort(self.moving_avg_sparsity[1:], direction='DESCENDING') + 1
            ],
            axis=0
        )
        conv2d_output_gate.channel_rank.assign(channel_rank)

        # weights = conv2d_output_gate.full_conv.get_weights()
        # new_weights = [weights[0][:, :, :, permutation]]
        # if len(weights) > 1:
        #     new_weights.append(weights[1][permutation])
        # if len(weights) > 2:
        #     raise RuntimeError(
        #         f'Unexpected number of weights: {len(weights)}. " \
        #         Expected kernel and bias weights only'
        #     )

        # conv2d_output_gate.full_conv.set_weights(new_weights)

        # self.moving_avg_sparsity.assign(
        #     self.moving_avg_sparsity.gather_nd(permutation[:, tf.newaxis])
        # )

class ChannelPermutateCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_freq):
        super().__init__()
        self.batch_count = 0
        self.update_freq = update_freq

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count >= self.update_freq:
            self._apply_permutation()
            self.batch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.batch_count = 0
        self._apply_permutation()

    def _apply_permutation(self):
        for layer in self.model.layers:
            if type(layer) is ChannelPermutate:
                conv2d_layer = self.model.get_layer(layer.conv2d_output_gate_name)
                layer.apply_permutation(conv2d_layer)

    # TODO allow (de)serialization
