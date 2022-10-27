import os
import tensorflow as tf
import tensorflow_datasets as tfds

def dataset(batch_size: int, preprocessor):
    train_ds = tfds.load('cifar10', split='train', data_dir=os.path.expanduser('~/dataset'))
    valid_ds = tfds.load('cifar10', split='test', data_dir=os.path.expanduser('~/dataset'))
        
    train_ds = train_ds.map(lambda dict: # type: ignore
        (
            preprocessor(tf.image.resize(dict['image'], (224, 224), tf.image.ResizeMethod.AREA)),
            tf.one_hot(dict['label'], 10)
        )
    ).shuffle(1000).batch(batch_size).prefetch(1)
    valid_ds = valid_ds.map(lambda dict: # type: ignore
        (
            preprocessor(tf.image.resize(dict['image'], (224, 224), tf.image.ResizeMethod.AREA)),
            tf.one_hot(dict['label'], 10)
        )
    ).shuffle(1000).batch(batch_size).prefetch(1)

    return train_ds, valid_ds
