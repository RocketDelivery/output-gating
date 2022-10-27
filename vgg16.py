import tensorflow as tf
import output_gate

def create(num_classes=1000, gate_threshold_target=0.0):
    base_model = tf.keras.applications.vgg16.VGG16()
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.layers.Input((224, 224, 3)))

    for layer in base_model.layers[1:-1]:
        # if type(layer) is tf.keras.layers.Conv2D:
        #     layer.activation = tf.keras.activations.linear
        #     new_model.add(layer)
        #     new_model.add(tf.keras.layers.BatchNormalization())
        #     new_model.add(tf.keras.layers.ReLU())
        #     continue
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
