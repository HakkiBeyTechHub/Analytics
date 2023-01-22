import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from matplotlib import pyplot as plt
import os
from pathlib import Path
import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import itertools
from collections import Counter
from sklearn.utils import class_weight
import numpy as np
from sklearn import metrics


tf.get_logger().setLevel("INFO")
h, w = 200, 200
batch_size = 32

TRAIN_DATAGEN = ImageDataGenerator(
    rescale=1.0 / 255.0,
    # vertical_flip=True,         # vertical transposition
    # horizontal_flip=True,       # horizontal transposition
    # rotation_range=90,          # random rotation at 90 degrees
    # height_shift_range=0.3,     # shift the height of the image 30%
    # brightness_range=[0.1, 0.9],
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

TEST_DATAGEN = ImageDataGenerator(rescale=1.0 / 255.0)

TRAIN_GENERATOR = TRAIN_DATAGEN.flow_from_directory(
    directory=Path("woods_dataset").joinpath("Train"),
    batch_size=batch_size,
    target_size=(h, w),
    seed=1,
)

VAL_GENERATOR = TEST_DATAGEN.flow_from_directory(
    Path("woods_dataset").joinpath("Test"),
    # subset="validation",
    seed=1,
    target_size=(h, w),
    batch_size=batch_size,
)



model = Sequential(
    [
        layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(h, w, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 5, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        # layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        # layers.Dense(128, activation='relu'),
        layers.Dense(12, activation="softmax"),
    ]
)


learning_rate = 1e-3
print(learning_rate)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: learning_rate * 10 ** (epoch / 20)
)
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


print(model.summary())


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

epochs = 150
class_weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(TRAIN_GENERATOR.classes),
    y=TRAIN_GENERATOR.classes,
)

train_class_weights = dict(enumerate(class_weights))
history = model.fit(
    TRAIN_GENERATOR,
    validation_data=VAL_GENERATOR,
    epochs=epochs,
    class_weight=train_class_weights,
    callbacks=[tensorboard_callback],
)


loss, accuracy = model.evaluate(VAL_GENERATOR)
print("Accuracy on test dataset:", accuracy)

X = []
y = []
for idx, k in enumerate(VAL_GENERATOR.as_numpy_iterator()):
    X.append(k[0])
    y.extend(k[1])

X = np.concatenate(X)

pred = model.predict(X)
pred = np.argmax(pred, axis=-1)
report = metrics.classification_report(y, pred)
print(report)
model.save("test_model")