import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow
import dogcat

if __name__ == '__main__':
    # dataset = tfds.load('cifar10', split='test', data_dir=os.path.expanduser('~/dataset'))
    # dataset = dataset.map(lambda dict: # type: ignore
    #     (
    #         tf.cast(tf.image.resize(dict['image'], (224, 224), tf.image.ResizeMethod.AREA), tf.uint8),
    #         tf.keras.applications.mobilenet.preprocess_input(
    #             tf.image.resize(dict['image'], (224, 224), tf.image.ResizeMethod.AREA) # type: ignore
    #         ),
    #         tf.one_hot(dict['label'], 10)
    #     )
    # ).batch(32)

    _, dataset = dogcat.create(
        32, 
        '/zfs/hdd0/dataset/dogcat/train', 
        '/zfs/hdd0/dataset/dogcat/test', 
        tf.keras.applications.vgg16.preprocess_input,
        preserve_input=True
    )

    sample_data = None
    images = None
    for images_take, sample_data_take, _ in dataset.take(1):
        images = images_take
        sample_data = sample_data_take

    app = QApplication([])
    win = MainWindow(os.path.expanduser('~/dataset/models/vgg16-dogcat-0.hdf5'), sample_data, images)
    win.show()
    app.exec_()
