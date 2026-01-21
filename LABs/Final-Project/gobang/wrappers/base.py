from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseWrapper(ABC):
    """
    Abstract base class for all player wrappers.
    All wrapper classes should inherit from this class and implement the required methods.
    """
    
    @abstractmethod
    def get_action(self, board_state: np.ndarray, **kwargs) -> Tuple[int, int]:
        """
        Get an action from the player given the current board state.
        
        Args:
            board_state: Current board state as numpy array
            **kwargs: Additional arguments that might be needed (e.g., temperature)
            
        Returns:
            Tuple of (row, col) representing the selected action
        """
        pass
    
    @abstractmethod
    def get_policy(self, board_state: np.ndarray) -> np.ndarray:
        """
        Get the policy distribution for the given board state.
        
        Args:
            board_state: Current board state as numpy array
            
        Returns:
            Policy distribution as numpy array
        """
        pass
    
    @abstractmethod
    def __call__(self, board_state: np.ndarray) -> np.ndarray:
        """
        Callable interface to get policy for a given board state.
        
        Args:
            board_state: Current board state
            
        Returns:
            Policy distribution
        """
        pass