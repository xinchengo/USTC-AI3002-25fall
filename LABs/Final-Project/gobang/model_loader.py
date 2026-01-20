import torch.nn as nn
from typing import *
from utils import *
import numpy as np
import torch
import os
from submission import GobangModel
from utils import device            # 设备配置(cpu/cuda/mps)

board_size = 12
bound = 5


# Load models using functions 'get_model' without passing any extra
# parameters, so that we can directly call get_model() in player.py and evaluator.py.


def get_model():
    # from submission import GobangModel
    # model = GobangModel(board_size=board_size, bound=bound)
    # model.load_state_dict(torch.load('model.pth'))
    # return model
    # 实例化模型架构
    model = GobangModel(board_size=board_size, bound=bound, use_deep=True)
    
    # 定义模型权重文件路径
    model_path = 'model.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"Warning: {model_path} not found. Trying to load from checkpoints...")
        # 本地 Deep 模型的文件名
        local_path = 'checkpoints_deep/model_2999.pth' 
        if os.path.exists(local_path):
             model.load_state_dict(torch.load(local_path, map_location=device))
    

    model.to(device)
    model.eval()
    
    return model


__all__ = ['get_model']
