import torch
from cnn.cnn_test import dummy_test
import numpy as np


def get_Yi(model, patch):
    with torch.no_grad():
        model(patch)
        return model.conv9.weight.values().reshape([400]).numpy()


def get_dummy_Yi():
    return dummy_test().reshape([400]).numpy()


def get_Y_hat(y: np.ndarray, operation: str):
    if operation == "max":
        return np.array(y).max(axis=0)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    else:
        raise Exception("The operation can be either mean or max")
