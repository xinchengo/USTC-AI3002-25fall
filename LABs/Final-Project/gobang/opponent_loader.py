import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch
import os
from submission import GobangModel
from utils import device

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_opponent(model_path: str = 'opponent.pth') -> nn.Module:
    # BEGIN YOUR CODE
    # from submission import GobangModel
    # opponent = GobangModel(board_size=board_size, bound=bound)
    # opponent.load_state_dict(torch.load('opponent.pth'))
    # return opponent
    model = GobangModel(board_size=board_size, bound=bound, use_deep=False)
    
    model_path = model_path
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # 本地测试回退逻辑
        local_path = 'checkpoints_baseline/model_2999.pth'
        if os.path.exists(local_path):
            model.load_state_dict(torch.load(local_path, map_location=device))
            print("Loaded local baseline opponent.")
        else:
            print("Warning: No opponent model found. Opponent will play RANDOMLY (initialized weights).")

    model.to(device)
    model.eval()
    return model
    # END YOUR CODE


__all__ = ['get_opponent']