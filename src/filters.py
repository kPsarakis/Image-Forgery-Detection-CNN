import torch
import numpy as np


def get_filters():

    filters = {}

    # 1st Order
    filters["1O1"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]))
    filters["1O2"] = torch.Tensor(np.rot90(filters["1O1"]).copy())
    filters["1O3"] = torch.Tensor(np.rot90(filters["1O2"]).copy())
    filters["1O4"] = torch.Tensor(np.rot90(filters["1O3"]).copy())
    filters["1O5"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]))
    filters["1O6"] = torch.Tensor(np.rot90(filters["1O5"]).copy())
    filters["1O7"] = torch.Tensor(np.rot90(filters["1O6"]).copy())
    filters["1O8"] = torch.Tensor(np.rot90(filters["1O7"]).copy())
    # 2nd Order
    filters["2O1"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]))
    filters["2O2"] = torch.Tensor(np.rot90(filters["2O1"]).copy())
    filters["2O3"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -2, 0, 0], [0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0]]))
    filters["2O4"] = torch.Tensor(np.rot90(filters["2O3"]).copy())
    # 3rd Order
    filters["3O1"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -3, 1, 0], [0, 0, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]))
    filters["3O2"] = torch.Tensor(np.rot90(filters["3O1"]).copy())
    filters["3O3"] = torch.Tensor(np.rot90(filters["3O2"]).copy())
    filters["3O4"] = torch.Tensor(np.rot90(filters["3O3"]).copy())
    filters["3O5"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 0]]))
    filters["3O6"] = torch.Tensor(np.rot90(filters["3O5"]).copy())
    filters["3O7"] = torch.Tensor(np.rot90(filters["3O6"]).copy())
    filters["3O8"] = torch.Tensor(np.rot90(filters["3O7"]).copy())
    # 3x3 SQUARE
    filters["3x3S"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0],
                                             [0, 0, 0, 0, 0]]))
    # 3x3 EDGE
    filters["3x3E1"] = torch.Tensor(np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0]]))
    filters["3x3E2"] = torch.Tensor(np.rot90(filters["3x3E1"]).copy())
    filters["3x3E3"] = torch.Tensor(np.rot90(filters["3x3E2"]).copy())
    filters["3x3E4"] = torch.Tensor(np.rot90(filters["3x3E3"]).copy())
    # 5X5 EDGE
    filters["5x5E1"] = torch.Tensor(np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
    filters["5x5E2"] = torch.Tensor(np.rot90(filters["5x5E1"]).copy())
    filters["5x5E3"] = torch.Tensor(np.rot90(filters["5x5E2"]).copy())
    filters["5x5E4"] = torch.Tensor(np.rot90(filters["5x5E3"]).copy())
    # 5x5 SQUARE
    filters["5x5S"] = torch.Tensor(np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                             [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]))

    return filters
