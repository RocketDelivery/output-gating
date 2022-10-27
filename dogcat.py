import os
import tensorflow as tf

@tf.function
def _process_data(file_path, size, preprocessor, preserve_input):
    image = tf.io.decode_jpeg(tf.io.read_file(file_path))
    image = tf.image.resize(image, size=size)
    # Image needs to be normalized before being fed into Resnet101 model
    image_preprocess = preprocessor(image)

    # Check if the file name is in the format (string)cat.(any number).jpg
    # See https://jex.im/regulex/#!flags=&re=%5E.*cat%5C.%5B0-9%5D%2B%5C.jpg%24
    is_cat = tf.strings.regex_full_match(file_path, '^.*cat\.[0-9]+\.jpg$')
    # Assign [0, 1] if cat, [1, 0] if otherwise
    onehot_label = tf.cond(
        is_cat, 
        lambda: tf.constant([0, 1], dtype=tf.float32), 
        lambda: tf.constant([1, 0], dtype=tf.float32))
    
    if preserve_input:
        return image, image_preprocess, onehot_label
    else:
        return image_preprocess, onehot_label

def dataset(batch_size, train_dir, valid_dir, preprocessor, preserve_input=False):
    train_ds = tf.data.Dataset.list_files(os.path.join(train_dir, '*.jpg'))
    valid_ds = tf.data.Dataset.list_files(os.path.join(valid_dir, '*.jpg'))
    
    train_ds = train_ds.shuffle(1000).map(lambda path: _process_data(path, (224, 224), preprocessor, preserve_input)).batch(batch_size).prefetch(1)
    valid_ds = valid_ds.shuffle(1000).map(lambda path: _process_data(path, (224, 224), preprocessor, preserve_input)).batch(batch_size).prefetch(1)

    return train_ds, valid_ds
