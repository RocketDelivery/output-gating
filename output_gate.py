import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Conv2DOutputGate(tf.keras.layers.Layer):
    def __init__(self, filters, cond_ratio, kernel_size=(3, 3), strides=1, padding='valid', activation=None, use_bias=True, 
                 gate_thresh_target=0, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 gate_bias_initializer='zeros', **kargs):
        if cond_ratio < 0 or cond_ratio > 1:
            raise ValueError('Conditional output ratio must be between 0 and 1')

        super().__init__(**kargs)
        self.filters = filters
        self.cond_ratio = cond_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.full_conv = tf.keras.layers.Conv2D(
            filters=filters, 
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding, 
            activation='linear', 
            use_bias=use_bias, 
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer)
        self.activation = activation
        self.gate_thresh_target = gate_thresh_target
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.gate_bias_initializer = gate_bias_initializer
        self.batchnorm_output = tf.keras.layers.BatchNormalization()
        self.batchnorm_metric = tf.keras.layers.BatchNormalization()
        # Note that epsilon will be initialized by a callback
        self.epsilon = self.add_weight(shape=(1,), initializer='zeros', trainable=False, name='epsilon', dtype=tf.float32)
        if type(self.activation) is str:
            self._activation_func = tf.keras.activations.get(self.activation)
        elif activation is not None:
            self._activation_func = activation
        else:
            self._activation_func = tf.keras.activations.linear
        
        # For visualization only. capture_output flag will be set by visualizer to cache intermediate value before ReLU
        self.capture_output = False
        self.intermediate_output = None

    def build(self, input_shape):               
        self.gate_thresh = self.add_weight(
            shape=(1,), 
            initializer=self.gate_bias_initializer, 
            trainable=True, 
            name='gate_threshold',
            dtype=tf.float32
        )

    def call(self, inputs, training=None):
        base_size = self.filters - int(self.filters * self.cond_ratio)

        full_output = self.full_conv(inputs)
        prune_metric = full_output[:, :, :, 0, tf.newaxis] # type: ignore
        full_output = self.batchnorm_output(full_output, training=training)
        prune_metric = self.batchnorm_metric(prune_metric, training=training)  

        base_output = full_output[..., :base_size] # type: ignore
        cond_output = full_output[..., base_size:] # type: ignore

        prune_metric -= self.gate_thresh
        prune_metric = tfp.math.clip_by_value_preserve_gradient(prune_metric, -10/self.epsilon, 10/self.epsilon)

        if training:
            cond_output = cond_output * (1 / (1 + tf.exp(-self.epsilon * prune_metric)))
        else:
            cond_output = tf.where(prune_metric > 0, cond_output, tf.zeros_like(cond_output))
        output = tf.concat([base_output, cond_output], axis=-1)

        if self.capture_output:
            self.intermediate_output = output.numpy() # type: ignore
        
        return self._activation_func(output), tf.pow(self.gate_thresh_target - self.gate_thresh, 2) # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'cond_ratio': self.cond_ratio,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'gate_thresh_target': self.gate_thresh_target,
            # 'kernel_initializer': self.kernel_initializer,
            # 'bias_initializer': self.bias_initializer,
            'gate_bias_initializer': self.gate_bias_initializer
        })
        return config

    def set_weights(self, weights):
        self.full_conv.set_weights(weights)
        # TODO handle cases where the weights are from Conv2DOutputGate instance

def replace_with_ogate(model, layer_indices):
    input = tf.keras.Input(shape=(224, 224, 3))
    cur_output = input
    gate_bias_diffs = []

    for index, layer in enumerate(model.layers):
        if index in layer_indices:
            if type(layer) is tf.keras.layers.Conv2D:
                params = layer_indices[index]
                sparsity_ratio = params.get('sparsity_ratio', None)
                if sparsity_ratio is None:
                    print(
                        f'Warning: sparsity ratio was not given for layer index {index}. Using default value 0.5'
                    )
                    sparsity_ratio = 0.5
                gate_threshold_target = params.get('gate_threshold_target', None)
                if gate_threshold_target is None:
                    print(
                        f'Warning: gate threshold target was not given for layer index {index}. Using default value 0'
                    )
                    gate_threshold_target = 0

                cur_output, gate_bias_diff = Conv2DOutputGate(
                    layer.filters, 
                    layer_indices[index]['sparsity_ratio'], 
                    layer.kernel_size, 
                    strides=layer.strides, # type: ignore
                    padding=layer.padding,
                    activation=layer.activation,
                    use_bias = layer.use_bias,
                    gate_thresh_target=gate_threshold_target,
                    kernel_initializer=layer.kernel_initializer, # type: ignore
                    bias_initializer=layer.bias_initializer, # type: ignore
                    name=layer.name + '_ogate'
                )(cur_output) # type: ignore
                gate_bias_diffs.append(gate_bias_diff)
                continue
            else:
                print(f'Warning: the layer at index {index} is not of a type replaceable with output gating')

        cur_output = layer(cur_output)

    new_model = tf.keras.Model(inputs=[input], outputs=[cur_output])
    gate_bias_diffs = tf.concat(gate_bias_diffs, axis=-1)
    gate_bias_diffs = 0.1 * tf.reduce_sum(gate_bias_diffs)
    new_model.add_loss(gate_bias_diffs)

    new_model.build((None, 224, 224, 3))

    for new_layer, base_layer in zip(new_model.layers[1:], model.layers[:]):
        new_layer.set_weights(base_layer.get_weights())
    
    return new_model
