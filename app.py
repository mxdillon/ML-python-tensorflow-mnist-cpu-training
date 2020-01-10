# coding=utf-8

from keras_onnx.keras2onnx.main import convert_keras
import onnx
import os
import sys
import json
from tf2onnx import loader
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

num_classes = 10
learning_rate = 0.003
num_epochs = 15

# Add a channels dimension to the data for batches
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

y_train = tf.one_hot(indices=y_train, depth=10)
y_test = tf.one_hot(indices=y_test, depth=10)

# Batch and shuffle the dataset
# train_ds = (tf.data.Dataset.from_tensor_slices(
#     (x_train, y_train)).shuffle(10000).batch(32))
# test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Step 2b: Define model architecture and compiles
batchX_placeholder = tf.placeholder(tf.float32, shape=[None, x_train[0].shape[0], x_train[0].shape[1], 1], name="input")
batchy_placeholder = tf.placeholder(tf.float32, shape=[None, num_classes])

# Step 2c: Setup Tensorboard logging
logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# define architecture
flatten = tf.layers.Flatten()
dense_1 = tf.layers.Dense(128, activation="relu")
dropout = tf.layers.Dropout(0.2)
softmax = tf.layers.Dense(10, activation="softmax")

# Define graph route
y_pred = flatten(batchX_placeholder)
y_pred = dense_1(y_pred)
y_pred = dropout(y_pred)
y_pred = softmax(y_pred)
y_pred = tf.identity(y_pred, name="output")

loss = tf.losses.softmax_cross_entropy(onehot_labels=batchy_placeholder, logits=y_pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(batchy_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_scalar = tf.summary.scalar("accuracy", accuracy)


init = tf.global_variables_initializer()
# batch_list = list(range(self.num_batches))

with tf.Session() as sess:
    sess.run(init)
    # writer = tf.summary.FileWriter(f'./{os.environ["TENSORBOARD_FOLDER_NAME"]}', sess.graph)

    for epoch in range(num_epochs):
        # random.shuffle(batch_list)
        # for batch in batch_list:
        _, train_accuracy, train_loss = sess.run((train, accuracy, loss),
                                                 feed_dict={batchX_placeholder: x_train,
                                                            batchy_placeholder: sess.run(y_train)})

        test_accuracy, test_loss = sess.run((accuracy, loss),
                                            feed_dict={batchX_placeholder: x_test,
                                                       batchy_placeholder: sess.run(y_test)})
        # writer.add_summary(train_accuracy, epoch)
        print(train_loss)
        print(train_accuracy)
        print(test_loss)
        print(test_accuracy)

    frozen_session = loader.freeze_session(sess, output_names=["output:0"])


# Only persist the model if we have passed our desired threshold
# if (train_loss > target_loss or train_accuracy < target_accuracy
#         or test_loss > target_loss or test_accuracy < target_accuracy):
#     sys.exit("Training failed to meet threshold")

# Step 4: Persist the trained model in ONNX format in the local file system along with any significant metrics

# Export the model to a SavedModel
with tf.gfile.GFile('model.pb', "wb") as f:
    f.write(frozen_session.SerializeToString())

# Convert SavedModel to ONNX format
command = f"python -m tf2onnx.convert --opset 11 --fold_const --verbose --input model.pb --output model.onnx --inputs input:0 --outputs output:0"
os.system(command)
# ^This currently has to be run using the command line tool as some arguments are not supported in the
# python API. See https://github.com/onnx/tensorflow-onnx#using-the-python-api for details.

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
