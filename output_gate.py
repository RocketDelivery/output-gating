import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class Conv2DOutputGate(tf.keras.layers.Layer):
    def __init__(self, filters, cond_ratio, kernel_size=(3, 3), strides=1, padding='valid', activation=None, use_bias=True, 
                 kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 **kargs):
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
            bias_initializer=bias_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(0.0005))

        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
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
        
        self.capture_output = False
        self.intermediate_output = None

        self.gate_thresh = self.add_weight(
                shape=(1,), 
                initializer=tf.keras.initializers.Constant(0), 
                trainable=False, 
                name='gate_threshold',
                dtype=tf.float32
            )

    def build(self, input_shape):               
        pass

    def call(self, inputs, training=None):
        base_size = self.filters - int(self.filters * self.cond_ratio)

        conv = self.full_conv
        full_output = conv(inputs, training=training)

        prune_metric = full_output[:, :, :, 0, tf.newaxis] # type: ignore
        full_output = self.batchnorm_output(full_output, training=training)
        prune_metric = self.batchnorm_metric(prune_metric, training=training)  

        base_output = full_output[..., :base_size] # type: ignore
        cond_output = full_output[..., base_size:] # type: ignore

        prune_metric -= self.gate_thresh

        @tf.custom_gradient
        def gate_func(metric):
            def grad(metric):
                sigmoid = tf.sigmoid(self.epsilon * metric)
                return sigmoid * (1 - sigmoid)

            return tf.where(metric > 0, tf.ones_like(metric), tf.zeros_like(metric)), grad

        cond_output = gate_func(prune_metric) * cond_output
        output = tf.concat([base_output, cond_output], axis=-1)

        if self.capture_output:
            self.intermediate_output = output.numpy() # type: ignore
        
        return self._activation_func(output) # type: ignore

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
        })
        return config

    def set_weights(self, weights):
        self.full_conv.set_weights(weights)
        # TODO handle cases where the weights are from Conv2DOutputGate instance

def replace_with_ogate(model, layer_indices):
    input = tf.keras.Input(shape=(224, 224, 3))
    cur_output = input
    weight_transfer_pairs = []
    cur_index = 1

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
                
                cur_output = Conv2DOutputGate(
                    layer.filters, 
                    layer_indices[index]['sparsity_ratio'], 
                    layer.kernel_size, 
                    strides=layer.strides, # type: ignore
                    padding=layer.padding,
                    activation=layer.activation,
                    use_bias=layer.use_bias,
                    kernel_initializer=layer.kernel_initializer, # type: ignore
                    bias_initializer=layer.bias_initializer, # type: ignore
                    name=layer.name + '_ogate'
                )(cur_output) # type: ignore
                weight_transfer_pairs.append((index, cur_index))
                cur_index += 1
                continue
            else:
                print(f'Warning: the layer at index {index} is not of a type replaceable with output gating')
        # Insert dropout layers
        elif layer.name in {'fc1', 'fc2'}:
            cur_output = tf.keras.layers.Dense(
                layer.units, 
                layer.activation, 
                layer.use_bias,
                kernel_regularizer=tf.keras.regularizers.l2(0.0005))(cur_output)
            weight_transfer_pairs.append((index, cur_index))
            cur_output = tf.keras.layers.Dropout(0.5)(cur_output)
            cur_index += 2
            continue
            
        cur_output = layer(cur_output)
        cur_index += 1

    new_model = tf.keras.Model(inputs=[input], outputs=[cur_output])

    new_model.build((None, 224, 224, 3))

    for base_index, new_index in weight_transfer_pairs:
        new_model.layers[new_index].set_weights(model.layers[base_index].get_weights())

    return new_model
