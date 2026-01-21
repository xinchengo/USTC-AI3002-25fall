"""
Alpha-Beta pruning baseline for Gobang.
Ported from https://github.com/lihongxun945/gobang
"""

from .board import Board
from .minmax import minmax
from .config import config

__all__ = ['Board', 'minmax', 'config']
