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

@tf.function
def _random_crop(image):
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

    new_shape = tf.shape(image)
    offsets = tf.cast(new_shape[0:2] - tf.constant([224, 224]), dtype=tf.float32) * tf.random.uniform((2,))
    offsets = tf.cast(offsets, dtype=tf.int32)
    return image[offsets[0]:offsets[0]+224, offsets[1]:offsets[1]+224] # type: ignore
    
@tf.function
def _random_horizontal_flip(image):
    if tf.random.uniform((1,)) > 0.5:
        image = image[..., :, ::-1, :]
    return image

def dataset(batch_size, preprocessor):
    train_ds, valid_ds = tfds.load('imagenet2012', split=['train', 'validation'], data_dir='/compas/benchmarks/imagenet/tfds')
    
    train_ds = train_ds.map(lambda dict: # type: ignore
        (
            preprocessor(_random_horizontal_flip(_random_crop(dict['image']))),
            tf.one_hot(dict['label'], 1000)
        )
    ).shuffle(1000).batch(batch_size).prefetch(1)
    
    # Disable validation set for now. We need to convert the model to fully convolutional model to perform validation
    # valid_ds = valid_ds.map(lambda dict: # type: ignore
    #     (
    #         preprocessor(dict['image']),
    #         tf.one_hot(dict['label'], 1000)
    #     )
    # ).batch(1).prefetch(1)

    return train_ds, None
