# coding=utf-8

import onnxruntime as rt
import keras2onnx
import onnx
import os
import sys
import json
from datetime import datetime
import tensorflow as tf

# Step 1: Set up target metrics for evaluating training

# Define a target loss metric to aim for
target_loss = 0.35
target_accuracy = 0.90

# Step 2: Perform training for the model

# Step 2a: Download and format training data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension to the data for batches
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Batch and shuffle the dataset
train_ds = (tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Step 2b: Define model architecture and compiles
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Step 2c: Setup Tensorboard logging
logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Step 2d: Train model
history = model.fit(train_ds, epochs=10, callbacks=[tensorboard_callback])

# Step 3: Evaluate model performance
train_loss, train_accuracy = (
    history.history["loss"][history.epoch[-1]],
    history.history["acc"][history.epoch[-1]],
)
test_loss, test_accuracy = model.evaluate(test_ds, verbose=2)

# Only persist the model if we have passed our desired threshold
if (train_loss > target_loss or train_accuracy < target_accuracy
        or test_loss > target_loss or test_accuracy < target_accuracy):
    sys.exit("Training failed to meet threshold")

# Step 4: Persist the trained model in ONNX format in the local file system along with any significant metrics
onnx_model = keras2onnx.convert_keras(model)
onnx.save_model(onnx_model, "model.onnx")

model = onnx.load("model.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

rt.set_default_logger_severity(0)
sess = rt.InferenceSession("model.onnx")

# Write metrics
if not os.path.exists("metrics"):
    os.mkdir("metrics")
with open("metrics/trainingloss.metric", "w+") as f:
    json.dump(str(train_loss), f)
with open("metrics/testloss.metric", "w+") as f:
    json.dump(str(test_loss), f)
with open("metrics/trainingaccuracy.metric", "w+") as f:
    json.dump(str(train_accuracy), f)
with open("metrics/testaccuracy.metric", "w+") as f:
    json.dump(str(test_accuracy), f)
