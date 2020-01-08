# ML-python-tensorflow-mnist-cpu-training

Quickstart project for training a MNIST classifier using TensorFlow on a CPU. Includes TensorBoard logging of training loss and training accuracy.

In accordance with MLOps principles, running `~app.py` will train a model and, if threshold metrics are passed, will save metrics to a `~.metrics/` folder and will convert the model to `.onnx` format, saving it as `~.model.onnx`.

Jenkins X requires the metrics and model to be saved in this format and the defined locations in order to promote the model to the service stage.
