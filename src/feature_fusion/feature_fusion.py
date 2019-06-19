import torch
import numpy as np


def get_Yi(model, patch):
    with torch.no_grad():
        model.eval()
        return model(patch)


def get_Y_hat(y: np.ndarray, operation: str):
    if operation == "max":
        return np.array(y).max(axis=0)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    else:
        raise Exception("The operation can be either mean or max")
