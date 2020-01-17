# ML-python-tensorflow-mnist-cpu-training

Quickstart project for training a MNIST classifier using TensorFlow on a CPU. Includes TensorBoard logging of training loss and training accuracy.

* In accordance with MLOps principles, running `requirements.txt` then `python app.py` will train a model and, if threshold metrics are passed, will convert the model to `.onnx` format, saving it as `.model.onnx`. 

* Additionally, metrics will be saved to a `.metrics/` folder.

* Tensorboard logs will also be saved in the `.metrics/` folder.

* Upon successful training, a Pull Request will automatically be made on the corresponding service project with the model and metrics folder being copied across.

* Jenkins X requires the metrics and model to be saved in this format and the defined locations in order to promote the model to the service stage.
