import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_model():
    # from submission import GobangModel
    # model = GobangModel(board_size=board_size, bound=bound)
    # model.load_state_dict(torch.load('model.pth'))
    # return model
    raise NotImplementedError("Not implemented!")


__all__ = ['get_model']
