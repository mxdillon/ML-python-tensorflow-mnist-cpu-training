#!/usr/bin/python
# # coding=utf-8


from src.train import TrainMNIST


# Step 1: Set up target metrics for evaluating training

# Define a target loss metric to aim for
target_loss = 0.35
target_accuracy = 0.90

# Step 2: Perform training for the model
model_trainer = TrainMNIST(batch_sz=32, epochs=1, target_loss=target_loss, target_accuracy=target_accuracy)
model_trainer.load_format_data()
model_trainer.define_model()
model_trainer.setup_tensorboard()
model_trainer.train_model()

# Step 3: Evaluate model performance
model_trainer.evaluate_performance()
# Only persist the model if we have passed our desired threshold
model_trainer.persistence_criteria()

# Step 4: Persist the trained model in ONNX format in the local file system along with any significant metrics
model_trainer.convert_save_onnx()
# Write metrics
model_trainer.save_metrics()
