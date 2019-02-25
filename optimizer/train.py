import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.python.keras.optimizers import Adam

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
num_channels = 1
num_classes = 10

# Data

tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data', one_hot=True)
tf.logging.set_verbosity(tf.logging.INFO)
validation_data = data.validation.images, data.validation.labels

# Model

model = Sequential()
model.add(InputLayer(input_shape=(img_size_flat,)))
model.add(Reshape(img_shape_full))
model.add(
    Conv2D(
        kernel_size=5, strides=1, filters=16, padding='same',
        activation=activation, name='layer_conv1'))
model.add(
    MaxPooling2D(pool_size=2, strides=2))
model.add(
    Conv2D(
        kernel_size=5, strides=1, filters=36, padding='same',
        activation=activation, name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))
model.add(Flatten())
for i in range(num_dense_layers):
    name = 'layer_dense_{0}'.format(i+1)
    model.add(
        Dense(
            num_dense_nodes,
            activation=activation,
            name=name))
model.add(Dense(num_classes, activation='softmax'))
optimizer = Adam(lr=learning_rate)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train

callbacks = [
    EarlyStopping(
        monitor='val_acc',
        baseline=0.5,
        patience=2),
    TensorBoard(
        log_dir='logs',
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
]

history = model.fit(
    x=data.train.images,
    y=data.train.labels,
    epochs=epochs,
    batch_size=128,
    validation_data=validation_data,
    callbacks=callbacks)
