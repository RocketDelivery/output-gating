import tensorflow as tf
import output_gate
import channel_permutate

def create_B0():
    conv_convert_names = {
        ('block1a_project_conv', 'block1a_project_bn', 'abs'),
        ('block2a_expand_conv', 'block2a_expand_activation', 'linear'),
        ('block2a_project_conv', 'block2a_project_bn', 'abs'),
        ('block2b_expand_conv', 'block2b_expand_activation', 'linear'),
        ('block2b_project_conv', 'block2b_project_bn', 'abs'),
        ('block3a_expand_conv', 'block3a_expand_activation', 'linear'),
        ('block3a_project_conv', 'block3a_project_bn', 'abs'),
        ('block3b_expand_conv', 'block3b_expand_activation', 'linear'),
        ('block3b_project_conv', 'block3b_project_bn', 'abs'),
        ('block4a_expand_conv', 'block4a_expand_activation', 'linear'),
        ('block4a_project_conv', 'block4a_project_bn', 'abs'),
        ('block4b_expand_conv', 'block4b_expand_activation', 'linear'),
        ('block4b_project_conv', 'block4b_project_bn', 'abs'),
        ('block4c_expand_conv', 'block4c_expand_activation', 'linear'),
        ('block4c_project_conv', 'block4c_project_bn', 'abs'),
        ('block5a_expand_conv', 'block5a_expand_activation', 'linear'),
        ('block5a_project_conv', 'block5a_project_bn', 'abs'),
        ('block5b_expand_conv', 'block5b_expand_activation', 'linear'),
        ('block5b_project_conv', 'block5b_project_bn', 'abs'),
        ('block5c_expand_conv', 'block5c_expand_activation', 'linear'),
        ('block5c_project_conv', 'block5c_project_bn', 'abs'),
        ('block6a_expand_conv', 'block6a_expand_activation', 'linear'),
        ('block6a_project_conv', 'block6a_project_bn', 'abs'),
        ('block6b_expand_conv', 'block6b_expand_activation', 'linear'),
        ('block6b_project_conv', 'block6b_project_bn', 'abs'),
        ('block6c_expand_conv', 'block6c_expand_activation', 'linear'),
        ('block6c_project_conv', 'block6c_project_bn', 'abs'),
        ('block6d_expand_conv', 'block6d_expand_activation', 'linear'),
        ('block6d_project_conv', 'block6d_project_bn', 'abs'),
        ('block7a_expand_conv', 'block7a_expand_activation', 'linear'),
        ('block7a_project_conv', 'block7a_project_bn', 'abs')
    }

    model = tf.keras.applications.efficientnet.EfficientNetB0()

    convert_name_mapping = {}
    for conv_layer_name, _, _ in conv_convert_names:
        conv2d_layer = model.get_layer(conv_layer_name)
        convert_name_mapping.update(
            {conv_layer_name: output_gate.Conv2DOutputGate(conv2d_layer, 0.5)})

    add_after_name_mapping = {}
    for conv_layer_name, add_after_name, channel_rank_mode in conv_convert_names:
        conv2d_ogate_layer = convert_name_mapping.get(conv_layer_name)
        add_after_name_mapping.update(
            {add_after_name: channel_permutate.ChannelPermutate(conv2d_ogate_layer, 0.99, channel_rank_mode)})

    model = output_gate.replace_with_ogate(
        model, 
        convert_name_mapping=convert_name_mapping,
        add_after_name_mapping=add_after_name_mapping
    )
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=[tf.keras.losses.CategoricalCrossentropy()], 
        metrics=['accuracy']
    )
    return model
