import os
import tf2onnx
import tensorflow as tf
import onnx
import onnxruntime as rt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

tf.keras.backend.set_floatx('float32')
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(train_ds, epochs=1)

tf.keras.experimental.export_saved_model(model, "./saved_model")
command = f"python -m tf2onnx.convert --opset 10 --fold_const --verbose --saved-model ./saved_model --output model.onnx"
os.system(command)

# load model for inference
model = onnx.load("model.onnx")

onnx.checker.check_model(model)

sess = rt.InferenceSession("model.onnx")
