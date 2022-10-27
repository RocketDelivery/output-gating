import os
import tensorflow as tf
import mobilenet_v1
import cifar10
import imagenet2012
import dogcat
import vgg16
from update_epsilon import UpdateEpsilon

BATCH_SIZE = 32
EPOCHS = 200

# tf.debugging.experimental.enable_dump_debug_info(
#     "logs/vgg-debug",
#     tensor_debug_mode="FULL_HEALTH",
#     circular_buffer_size=-1)

model = vgg16.create(num_classes=1000, gate_threshold_target=0)
model.summary()
# train_ds, valid_ds = dogcat.create(
#     BATCH_SIZE, 
#     '/zfs/hdd0/dataset/dogcat/train', 
#     '/zfs/hdd0/dataset/dogcat/test', 
#     tf.keras.applications.vgg16.preprocess_input
# )
train_ds, valid_ds = imagenet2012.dataset(BATCH_SIZE, tf.keras.applications.vgg16.preprocess_input);

my_callbacks = [
    # Stop the training when the result hasn't improved in 5 epochs
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_accuracy'),
    # Save the model file for every epoch
    tf.keras.callbacks.ModelCheckpoint(filepath='/zfs/hdd0/dataset/tmp/train-{epoch:02d}.hdf5'),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.2, 
        patience=5, 
        model='max', 
        min_lr=0.00001), # type: ignore
    UpdateEpsilon(1, 8, 20),
    tf.keras.callbacks.BackupAndRestore(os.path.abspath('backup'), save_freq=10000), # type: ignore
    # Tensorboard logging
    tf.keras.callbacks.TensorBoard(log_dir='logs/vgg16-imagenet-0'),
]

model.fit(train_ds, validation_data=valid_ds, epochs=200, callbacks=my_callbacks)
