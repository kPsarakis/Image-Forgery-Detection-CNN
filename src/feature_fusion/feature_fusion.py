import math

import torch
import numpy as np


def get_yi(model, patch):
    """
    Returns the patch's feature representation
    :param model: The pre-trained CNN object
    :param patch: The patch
    :returns: The 400-D feature representation of the patch
    """
    with torch.no_grad():
        model.eval()
        return model(patch)


class WrongOperationOption(Exception):
    pass


def get_y_hat(y: np.ndarray, operation: str):
    """
    Fuses the image's patches feature representation
    :param y: The network object
    :param operation: Either max or mean for the pooling operation
    :returns: The final 400-D feature representation of the entire image
    """
    if operation == "max":
        return np.array(y).max(axis=0, initial=-math.inf)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    else:
        raise WrongOperationOption("The operation can be either mean or max")
