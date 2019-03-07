import tensorflow as tf

from tensorflow.python.keras import callbacks
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers

import util

# Hyperparameters

activation = 'relu'
num_dense_layers = 1
num_dense_nodes = 16
learning_rate = 1e-5
epochs = 3

# Model params

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
img_shape_full = (img_size, img_size, 1)
num_classes = 10

# Data

tf.logging.set_verbosity(tf.logging.WARN)
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data', one_hot=True)
tf.logging.set_verbosity(tf.logging.INFO)
validation_data = data.validation.images, data.validation.labels

# Model

model = models.Sequential()
model.add(layers.InputLayer(input_shape=(img_size_flat,)))
model.add(layers.Reshape(img_shape_full))
model.add(layers.Conv2D(
    kernel_size=5,
    strides=1,
    filters=16,
    padding='same',
    activation=activation,
    name='layer_conv1'))
model.add(layers.MaxPooling2D(pool_size=2, strides=2))
model.add(layers.Conv2D(
    kernel_size=5,
    strides=1,
    filters=36,
    padding='same',
    activation=activation,
    name='layer_conv2'))
model.add(layers.MaxPooling2D(pool_size=2, strides=2))
model.add(layers.Flatten())
for i in range(num_dense_layers):
    name = 'layer_dense_{0}'.format(i+1)
    model.add(layers.Dense(
        num_dense_nodes,
        activation=activation,
        name=name))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(
    optimizer=optimizers.Adam(lr=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

util.save_model_summary(model)

# Train

callbacks = [
    callbacks.EarlyStopping(
        monitor='val_acc',
        patience=2),
    callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False),
    callbacks.ModelCheckpoint(
        filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
]

model.fit(
    x=data.train.images,
    y=data.train.labels,
    epochs=epochs,
    batch_size=128,
    validation_data=validation_data,
    callbacks=callbacks)
