import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# tensorflow_transform won't install. Leave it out for now
# @tf.function
# def _rgb_shift(image):
#     pca_vec = tft.pca(
#         tf.reshape(image, (-1, tf.shape(image)[-1])), 
#         output_dim=3)
#     scale = tf.random.normal((3,), mean=0, stddev=0.1)
#     image += pca_vec[tf.newaxis, :, 0] * scale[0]
#     image += pca_vec[tf.newaxis, :, 1] * scale[1]
#     image += pca_vec[tf.newaxis, :, 2] * scale[2]
#     return image

def _resize_min(image):
    orig_shape = tf.cast(tf.shape(image), tf.float32)
    height, width = (orig_shape[0], orig_shape[1]) # type: ignore

    # Rows (height)
    if height > width:
        image = tf.image.resize(
            image, 
            (int(height * 256.0 / width), 256),
            tf.image.ResizeMethod.AREA)
    # Columns (width)
    else:
        image = tf.image.resize(
            image, 
            (256, int(width * 256.0 / height)),
            tf.image.ResizeMethod.AREA)
    
    return image

# @tf.function
def _random_crop(image):
    image = _resize_min(image)
    new_shape = tf.shape(image)
    offsets = tf.cast(new_shape[0:2] - tf.constant([224, 224]), dtype=tf.float32) * tf.random.uniform((2,))
    offsets = tf.cast(offsets, dtype=tf.int32)
    return image[offsets[0]:offsets[0]+224, offsets[1]:offsets[1]+224] # type: ignore
    
# @tf.function
def _random_horizontal_flip(image):
    if tf.random.uniform((1,)) > 0.5:
        image = image[..., :, ::-1, :]
    return image

# @tf.function
def _preprocess(image, label, preprocessor):
    image = _random_crop(image)
    image = _random_horizontal_flip(image)
    image = preprocessor(image)
    return image, tf.one_hot(label, 1000)

# @tf.function
def _preprocess_valid(image, label, preprocessor):
    image = _resize_min(image)
    image = preprocessor(image)
    return image, tf.one_hot(label, 1000)

def dataset(batch_size, preprocessor):
    train_ds, valid_ds = tfds.load('imagenet2012', split=['train', 'validation'], as_supervised=True)
    
    train_ds = train_ds.map(
        lambda image, label: 
            _preprocess(image, label, preprocessor)) \
        .shuffle(1000).batch(batch_size).prefetch(1)
    valid_ds = valid_ds.map(
        lambda image, label: 
            _preprocess_valid(image, label, preprocessor)) \
        .batch(1).prefetch(1)

    return train_ds, valid_ds
