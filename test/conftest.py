#!/usr/bin/python
# # coding=utf-8

import pytest
from src.train import TrainMNIST


@pytest.fixture(scope='session')
def train_class():
    return TrainMNIST(batch_sz=64, epochs=1, target_loss=0.5, target_accuracy=0.80)


@pytest.fixture(scope='session')
def get_data(train_class):
    train_class.load_format_data()
    return train_class


@pytest.fixture(scope='session')
def get_model(get_data):
    get_data.define_model()
    return get_data


@pytest.fixture(scope='session')
def log_tb(get_model):
    get_model.setup_tensorboard()
    return get_model


@pytest.fixture(scope='session')
def train(log_tb):
    log_tb.train_model()
    return log_tb


@pytest.fixture(scope='session')
def evaluate(train):
    train.evaluate_performance()
    return train
