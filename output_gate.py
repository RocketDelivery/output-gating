import tensorflow as tf


class Conv2DOutputGate(tf.keras.layers.Layer):
    def __init__(self, conv2d_base, cond_ratio):
        if cond_ratio < 0 or cond_ratio > 1:
            raise ValueError('Conditional output ratio must be between 0 and 1')

        super().__init__(name=conv2d_base.name + '_ogate')

        self.cond_ratio = cond_ratio
        self.full_conv = conv2d_base
        self.activation = conv2d_base.activation

        conv2d_base.activation = tf.keras.activations.linear
        self.batchnorm_metric = tf.keras.layers.BatchNormalization()
        # Note that epsilon will be initialized by a callback
        self.epsilon = self.add_weight(
            shape=(1,), 
            initializer=tf.keras.initializers.Constant(1), 
            trainable=False, 
            name='epsilon', 
            dtype=tf.float32
        )
        self.channel_rank = self.add_weight(
            shape=(conv2d_base.filters,),
            trainable=False,
            name='channel_rank',
            dtype=tf.int32
        )
        self.channel_rank.assign(tf.range(0, conv2d_base.filters,  dtype=tf.int32))

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
        base_size = self.full_conv.filters - int(self.full_conv.filters * self.cond_ratio)

        full_output = self.full_conv(inputs, training=training)
        prune_metric = full_output[:, :, :, 0, tf.newaxis] # type: ignore
        prune_metric = self.batchnorm_metric(prune_metric, training=training)  
        
        prune_metric -= self.gate_thresh

        # @tf.custom_gradient
        # def gate_func(metric):
        #     def grad(metric):
        #         sigmoid = tf.sigmoid(self.epsilon * metric)
        #         return sigmoid * (1 - sigmoid)

        #     return tf.where(metric > 0, tf.ones_like(metric), tf.zeros_like(metric)), grad

        full_output_ranked = tf.gather(full_output, self.channel_rank, axis=-1)
        if training:
            full_output_ranked = tf.concat(
                [
                    full_output_ranked[..., :base_size], 
                    tf.sigmoid(self.epsilon * prune_metric) * full_output_ranked[..., base_size:]
                ], 
                axis=-1)
        else:
            full_output_ranked = tf.concat(
                [
                    full_output_ranked[..., :base_size], 
                    tf.where(
                        prune_metric > 0, 
                        tf.ones_like(prune_metric), 
                        tf.zeros_like(prune_metric)
                    ) * full_output_ranked[..., base_size:]
                ], 
                axis=-1)

        channel_rank_undo = tf.argsort(self.channel_rank, direction='ASCENDING')
        full_output = tf.gather(full_output_ranked, channel_rank_undo, axis=-1)

        # output = tf.concat([base_output, cond_output], axis=-1)

        if self.capture_output:
            self.intermediate_output = output.numpy() # type: ignore
        
        return self.activation(full_output) # type: ignore

    def get_config(self):
        config = super().get_config()
        config.update({
            'cond_ratio': self.cond_ratio,
            'activation': self.activation
        })
        return config

    def set_weights(self, weights):
        self.full_conv.set_weights(weights)
        # TODO handle cases where the weights are from Conv2DOutputGate instance


def replace_with_ogate(model, convert_name_mapping={}, add_after_name_mapping={}):
    # Based on the code from https://stackoverflow.com/a/54517478

    # Auxiliary dictionary to describe the network graph
    network_dict = { 'input_layers_of': {}, 'new_output_tensor_of': {} }

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            cur_layer_name = node.outbound_layer.name
            if cur_layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                    { cur_layer_name: [layer.name] }
                )
            else:
                network_dict['input_layers_of'][cur_layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
        { model.layers[0].name: model.input }
    )
    
    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        layer_output = None
        if layer.name in convert_name_mapping:
            new_layer = convert_name_mapping.get(layer.name)
            layer_output = new_layer(layer_input)                
        elif layer.name in add_after_name_mapping:
            new_layer = add_after_name_mapping.get(layer.name)
            layer_output = new_layer(layer(layer_input))
        else:
            if layer.name == 'tf.math.truediv' or layer.name == 'tf.math.truediv_1':
                layer_output = tf.truediv(layer_input, tf.constant([2.0897, 2.1129, 2.1082], dtype=tf.float32))
            else:
                layer_output = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: layer_output})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names: # type: ignore
            model_outputs.append(layer_output)

    return tf.keras.Model(inputs=model.inputs, outputs=model_outputs)
