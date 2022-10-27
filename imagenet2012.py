from matplotlib.rcsetup import validate_dashlist
import tensorflow as tf
import tensorflow_datasets as tfds

def dataset(batch_size, preprocessor):
    train_ds, valid_ds = tfds.load('imagenet2012', split=['train', 'validation'], data_dir='/zfs/hdd0/dataset')
    
    train_ds = train_ds.map(lambda dict: # type: ignore
        (
            preprocessor(tf.image.resize(dict['image'], (224, 224), tf.image.ResizeMethod.AREA)),
            tf.one_hot(dict['label'], 1000)
        )
    ).shuffle(1000).batch(batch_size).prefetch(1)
    valid_ds = valid_ds.map(lambda dict: # type: ignore
        (
            preprocessor(tf.image.resize(dict['image'], (224, 224), tf.image.ResizeMethod.AREA)),
            tf.one_hot(dict['label'], 1000)
        )
    ).shuffle(1000).batch(batch_size).prefetch(1)

    return train_ds, valid_ds
