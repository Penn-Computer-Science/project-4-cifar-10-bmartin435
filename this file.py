import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
sns.countplot(x=y_train)
plt.show()

print("Any NaN Training: ", np.isnan(x_train).any())
print("Any NaN Training: ", np.isnan(x_test).any())

input_shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train /255.0

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test /255.0

#convert tables to one hot
y_train = tf.one_hot(y_train.astype(np.int32), depth = 10)
y_test = tf.one_hot(y_test.astype(np.int32), depth = 10)

#show an example from MNIST
plt.imshow(x_train[100][:,:,0])
plt.show()

#time for sauce
batch_size = 128
num_classes = 10
epochs = 5

#building time
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(32, (5,5), padding="same", activation="relu", input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3,3), padding="same", activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2d(64, (3,3), padding="same", activation="relu", input_shape=input_shape),
        tf.keras.layers.Conv2d(64, (3,3), padding="same", activation="relu", input_shape=input_shape),
        tf.keras.layers.dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer=tf.keras.optimizers.RMSprop(epsilon=1e-08), loss="categorical_crossentropy", metrics=["acc"])