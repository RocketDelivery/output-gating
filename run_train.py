import os
import tensorflow as tf
import imagenet2012
import vgg16
from update_epsilon import UpdateEpsilon

BATCH_SIZE = 32 * 4
EPOCHS = 200

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = vgg16.create(num_classes=1000, gate_threshold_target=0)
model.summary()

train_ds, _ = imagenet2012.dataset(BATCH_SIZE, tf.keras.applications.vgg16.preprocess_input)

my_callbacks = [
    # Stop the training when the result hasn't improved in 10 epochs
    # Disable it for now because validation needs to be done separately
    # tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_accuracy'),

    # Save the model file for every epoch
    tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/train-{epoch:02d}.hdf5'),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy', 
        factor=0.1, 
        patience=5, 
        model='max', 
        min_lr=0.000001), # type: ignore

    UpdateEpsilon(4, 4, 40),

    tf.keras.callbacks.BackupAndRestore(os.path.abspath('backup'), save_freq=4000), # type: ignore

    # Tensorboard logging
    tf.keras.callbacks.TensorBoard(log_dir='logs/vgg16-imagenet'),
]

model.fit(train_ds, epochs=EPOCHS, callbacks=my_callbacks)
