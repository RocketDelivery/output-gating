import os
import tensorflow as tf
import imagenet2012
import vgg16
import efficientnet
from update_epsilon import UpdateEpsilon
from channel_permutate import ChannelPermutateCallback

BATCH_SIZE = 64
EPOCHS = 200

# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
model = efficientnet.create_B0()
model.summary()

train_ds, valid_ds = imagenet2012.dataset(BATCH_SIZE, tf.keras.applications.efficientnet.preprocess_input)

my_callbacks = [
    ChannelPermutateCallback(update_freq=1),

    # Stop the training when the result hasn't improved in 10 epochs
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_accuracy'),

    # Save the model file for every epoch
    tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/train-{epoch:02d}.hdf5'),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy', 
        factor=0.1, 
        patience=5, 
        model='max', 
        min_lr=0.000001), # type: ignore

    UpdateEpsilon(4, 4, 40),

    tf.keras.callbacks.BackupAndRestore(os.path.abspath('backup')), # type: ignore

    # Tensorboard logging
    tf.keras.callbacks.TensorBoard(log_dir='logs/efficientnet-imagenet'),
]

model.fit(train_ds, epochs=EPOCHS, callbacks=my_callbacks)
