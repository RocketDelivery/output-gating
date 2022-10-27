import tensorflow as tf

from output_gate import Conv2DOutputGate

def create(sparsity_ratio, gate_threshold):
    cur_replace = 0
    start_replace = 1
    end_replace = 2

    input = tf.keras.Input(shape=(224, 224, 3))

    base_model = tf.keras.applications.mobilenet.MobileNet()
    base_model.build(input_shape=(None, 224, 224, 3))

    gate_bias_diffs = []
    cur_output = input
    for index, layer in enumerate(base_model.layers[1:-3]):
        # base_model is enumerated from index 1, so add 1 to obtain correct layer index
        index += 1
        if type(layer) is tf.keras.layers.Conv2D:
            cur_replace += 1
            if cur_replace > start_replace and cur_replace <= end_replace:
                cur_output, gate_bias_diff = Conv2DOutputGate(
                    layer.filters, 
                    sparsity_ratio, 
                    layer.kernel_size, 
                    padding=layer.padding.upper(),
                    gate_thresh_target=gate_threshold, 
                    name=layer.name + '_ogate',
                )(cur_output) # type: ignore
                gate_bias_diffs.append(gate_bias_diff)
                continue
        
        cur_output = layer(cur_output)

    cur_output = tf.keras.layers.Conv2D(10, 1)(cur_output)
    cur_output = tf.keras.layers.Reshape((10,))(cur_output)
    cur_output = tf.keras.layers.Softmax()(cur_output)
    
    gate_bias_diffs = tf.concat(gate_bias_diffs, axis=-1)
    gate_bias_diffs = 0.1 * tf.reduce_sum(gate_bias_diffs)

    new_model = tf.keras.Model(inputs=[input], outputs=[cur_output])
    new_model.add_loss(gate_bias_diffs)

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=[tf.keras.losses.CategoricalCrossentropy()], 
        metrics=['accuracy']
    )

    new_model.build((None, 224, 224, 3))

    for ogate_layer, base_layer in zip(new_model.layers[1:-3], base_model.layers[1:-3]):
        ogate_layer.set_weights(base_layer.get_weights())

    return new_model
