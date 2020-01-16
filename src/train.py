#!/usr/bin/python
# # coding=utf-8

import os
import sys
import json
import onnx
import keras2onnx
import tensorflow as tf
from datetime import datetime


class TrainMNIST:
    """Train a simple Multi-Layer Percpetron network to classify MNIST images. Convert save model to .onnx format if
    training thresholds are met."""

    def __init__(self, batch_sz: int, epochs: int, target_loss: float, target_accuracy: float):
        """Instantiate the trainer class.

        :param batch_sz: size of batches to train the data in
        :type batch_sz: int
        :param epochs: [description]
        :type epochs: int
        :param target_loss: [description]
        :type target_loss: float
        :param target_accuracy: [description]
        :type target_accuracy: float
        """
        self.batch_sz = batch_sz
        self.epochs = epochs
        self.target_loss = target_loss
        self.target_accuracy = target_accuracy

        self.train_ds = None
        self.test_ds = None
        self.model = None
        self.tensorboard_callback = None
        self.history = None
        self.train_loss = None
        self.train_accuracy = None
        self.test_loss = None
        self.test_accuracy = None

    def load_format_data(self):
        """Download and format mnist dataset for training a classifier.

        Scale the pixel intensities so that they lie in the range [0,1] then batches the data according to the user defined
        input.
        :return: training and testing datasets (tensorflow objects)
        :rtype: tuple
        """
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Add a channels dimension to the data for batches
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]

        # Batch and shuffle the dataset
        self.train_ds = (tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(self.batch_sz))
        self.test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch_sz)

    def define_model(self):
        """Define model architecture and compile model."""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ])

        self.model.compile(optimizer="adam",
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

    def setup_tensorboard(self):
        """Setup tensorboard logging. If the logs are to be carried across to the service repository upon successful
        training, they must be saved within the './metrics/' folder."""
        logdir = "./metrics/tensorboard_logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, batch_size=self.batch_sz)

    def train_model(self):
        """Train the model."""
        self.history = self.model.fit(self.train_ds, epochs=self.epochs, callbacks=[self.tensorboard_callback])

    def evaluate_performance(self):
        """Evaluate model performance. Calculate the train loss and accuracy for the final epoch and calculate the test
        loss and accuracy over the whole test set."""
        self.train_loss, self.train_accuracy = (
            self.history.history["loss"][self.history.epoch[-1]],
            self.history.history["acc"][self.history.epoch[-1]],
        )

        self.test_loss, self.test_accuracy = self.model.evaluate(self.test_ds, verbose=2)

    def persistence_criteria(self):
        """Check that trained model meets the user-defined accpetance criteria for model performance. If not, quit."""
        if (self.train_loss > self.target_loss or self.train_accuracy < self.target_accuracy
                or self.test_loss > self.target_loss or self.test_accuracy < self.target_accuracy):
            sys.exit("Training failed to meet threshold")

    def convert_save_onnx(self):
        """Convert model to onnx, save onnx model as '.model.onnx' in local directory. This location is specified by
        Jenkins-x which is why it is hard coded in this method."""
        onnx_model = keras2onnx.convert_keras(self.model)
        onnx.save_model(onnx_model, "model.onnx")

    def save_metrics(self):
        """Save performance metrics in json format to './metrics/' folder. Again, this location is specified by
        Jenkins-x. The whole metrics folder will be copied into the service project upon successful completion of
        model training.
        """
        if not os.path.exists("metrics"):
            os.mkdir("metrics")
        with open("metrics/trainingloss.metric", "w+") as f:
            json.dump(str(self.train_loss), f)
        with open("metrics/testloss.metric", "w+") as f:
            json.dump(str(self.test_loss), f)
        with open("metrics/trainingaccuracy.metric", "w+") as f:
            json.dump(str(self.train_accuracy), f)
        with open("metrics/testaccuracy.metric", "w+") as f:
            json.dump(str(self.test_accuracy), f)
