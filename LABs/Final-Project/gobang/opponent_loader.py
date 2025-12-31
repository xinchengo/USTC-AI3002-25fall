import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_opponent():
    # BEGIN YOUR CODE
    # from submission import GobangModel
    # opponent = GobangModel(board_size=board_size, bound=bound)
    # opponent.load_state_dict(torch.load('opponent.pth'))
    # return opponent
    raise NotImplementedError("Not implemented!")
    # END YOUR CODE


__all__ = ['get_opponent']
