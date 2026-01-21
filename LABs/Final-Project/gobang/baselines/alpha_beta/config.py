"""
Configuration for alpha-beta pruning.
Ported from https://github.com/lihongxun945/gobang
"""


class Config:
    """Configuration class for alpha-beta pruning algorithm."""
    
    def __init__(self):
        self.enable_cache = True  # Whether to enable caching
        self.points_limit = 20  # Maximum nodes to search per level
        self.only_in_line = False  # Only search points in a line (optimization)
        self.inline_count = 4  # Recent moves count for inline detection
        self.in_line_distance = 5  # Maximum distance for inline detection


config = Config()
