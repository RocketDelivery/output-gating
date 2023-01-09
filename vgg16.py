import tensorflow as tf
import output_gate

def create(num_classes=1000, gate_threshold_target=0.0):
    base_model = tf.keras.applications.vgg16.VGG16()
    
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.layers.Input((224, 224, 3)))

    for layer in base_model.layers[1:-1]:
        new_model.add(layer)

    if num_classes != 1000:
        new_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    else:
        new_model.add(base_model.layers[-1])

    new_model = output_gate.replace_with_ogate(
        new_model, 
        {
            0:  {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            1:  {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            3:  {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            4:  {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            6:  {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            7:  {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            8:  {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            10: {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            11: {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            12: {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            14: {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            15: {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}, 
            16: {'sparsity_ratio': 0.5, 'gate_threshold_target': gate_threshold_target}
        })
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=[tf.keras.losses.CategoricalCrossentropy()], 
        metrics=['accuracy']
    )

    return new_model


def to_fully_conv(model):
    new_model = tf.keras.Sequential()
    input_layer = tf.keras.layers.InputLayer(input_shape=(None, None, 3), name="input_new")
    new_model.add(input_layer)

    f_dim = [0, 0, 0, 0]
    new_layer = None
    weights = []
    index = 0
    layer_state = 0

    for layer in model.layers:
        if "Flatten" in str(layer):
            layer_state = 1
            f_dim = layer.input_shape
            continue
        elif "Dense" in str(layer):
            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W, b = layer.get_weights()

            if layer_state == 1:
                shape = (f_dim[1], f_dim[2], f_dim[3], output_dim)
                new_W = tf.reshape(W, shape)
                new_layer = tf.keras.layers.Conv2D(output_dim, (f_dim[1], f_dim[2]), strides=(1, 1), activation=layer.activation, padding='valid')
            else:
                shape = (1, 1, input_shape[1], output_dim)
                new_W = tf.reshape(W, shape)
                new_layer = tf.keras.layers.Conv2D(output_dim, (1, 1), strides=(1, 1), activation='relu' if layer_state == 2 else 'linear', padding='valid')
            
            layer_state += 1
            weights.append((index, new_W, b))
        else:
            new_layer = layer

        new_model.add(new_layer)
        index += 1

    for index, W, b in weights:
        new_model.layers[index].set_weights([W, b])

    return new_model
