"""
Minimax Engine for Gobang - Exact translation from lihongxun945/gobang minmax.js
Implements iterative deepening minimax with alpha-beta pruning, VCT and VCF
"""

from typing import List, Tuple, Optional, Dict, Any
from .evaluator import BoardEvaluator, FIVE, FOUR


MAX = 1000000000

# Cache statistics
cache_hits = {
    'search': 0,
    'total': 0,
    'hit': 0
}

ONLY_THREE_THRESHOLD = 6


class Cache:
    """FIFO cache - exact translation from cache.js"""
    
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.cache = []  # FIFO queue of keys
        self.map = {}  # key -> value mapping
    
    def get(self, key: int) -> Optional[Dict[str, Any]]:
        """Get value for a key"""
        if key in self.map:
            return self.map[key]
        return None
    
    def put(self, key: int, value: Dict[str, Any]):
        """Set or insert a value"""
        if len(self.cache) >= self.capacity:
            oldest_key = self.cache.pop(0)  # Remove oldest key
            del self.map[oldest_key]  # Delete from map
        
        if key not in self.map:
            self.cache.append(key)  # Add new key to cache
        self.map[key] = value  # Update or set key-value
    
    def has(self, key: int) -> bool:
        """Check if key exists in cache"""
        return key in self.map


# Global cache shared by minmax, vct, and vcf
cache = Cache()


def factory(only_three: bool = False, only_four: bool = False):
    """
    Factory function to create minimax variants.
    
    Args:
        only_three: VCT mode (only three and four patterns)
        only_four: VCF mode (only four patterns)
    
    Returns:
        Helper function configured for the specified mode
    """
    
    def helper(evaluator: BoardEvaluator, role: int, depth: int, c_depth: int = 0,
               path: List[Tuple[int, int]] = None, alpha: float = -MAX, beta: float = MAX
               ) -> Tuple[float, Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Minimax helper with iterative deepening and alpha-beta pruning.
        
        Args:
            evaluator: Board evaluator
            role: Current player (1 or -1)
            depth: Total search depth
            c_depth: Current search depth
            path: Current path of moves
            alpha: Alpha value for pruning
            beta: Beta value for pruning
        
        Returns:
            Tuple of (value, move, best_path)
        """
        if path is None:
            path = []
        
        cache_hits['search'] += 1
        
        # Terminal condition
        if c_depth >= depth or evaluator.is_game_over():
            return evaluator.evaluate(role), None, path.copy()
        
        # Check cache
        hash_value = evaluator.hash()
        prev = cache.get(hash_value)
        if prev and prev['role'] == role:
            # Can use cache if it's a winning position or has sufficient depth
            # and mode matches (only_three/only_four)
            if ((abs(prev['value']) >= FIVE or prev['depth'] >= depth - c_depth) and
                prev['only_three'] == only_three and prev['only_four'] == only_four):
                cache_hits['hit'] += 1
                return prev['value'], prev['move'], path + prev['path']
        
        value = -MAX
        move = None
        best_path = path.copy()
        best_depth = 0
        
        # Get valuable moves
        points = evaluator.get_moves(role, c_depth,
                                    only_three or c_depth > ONLY_THREE_THRESHOLD,
                                    only_four)
        
        if c_depth == 0:
            print(f'points: {points}')
        
        if not points:
            return evaluator.evaluate(role), None, path.copy()
        
        # Iterative deepening - only search even depths
        for d in range(c_depth + 1, depth + 1):
            # Only search even depths (己方能赢的解)
            if d % 2 != 0:
                continue
            
            break_all = False
            
            for point in points:
                evaluator.move(point[0], point[1], role)
                new_path = path + [point]
                
                current_value, current_move, current_path = helper(
                    evaluator, -role, d, c_depth + 1, new_path, -beta, -alpha
                )
                current_value = -current_value
                
                evaluator.undo(point[0], point[1])
                
                # During iterative deepening, only accept winning moves or final depth
                # Reason: Non-winning evaluations are inaccurate at shallow depths
                if current_value >= FIVE or d == depth:
                    # For losing positions, choose the longest path (struggle)
                    if (current_value > value or
                        (current_value <= -FIVE and value <= -FIVE and
                         len(current_path) > best_depth)):
                        value = current_value
                        move = point
                        best_path = current_path
                        best_depth = len(current_path)
                
                alpha = max(alpha, value)
                
                # Break if we found a winning move
                if alpha >= FIVE:
                    break_all = True
                    break
                
                # Alpha-beta pruning
                if alpha >= beta:
                    break
            
            if break_all:
                break
        
        # Cache the result
        if ((c_depth < ONLY_THREE_THRESHOLD or only_three or only_four) and
            (not prev or prev['depth'] < depth - c_depth)):
            cache_hits['total'] += 1
            cache.put(hash_value, {
                'depth': depth - c_depth,  # Remaining depth
                'value': value,
                'move': move,
                'role': role,
                'path': best_path[c_depth:],  # Remaining path
                'only_three': only_three,
                'only_four': only_four
            })
        
        return value, move, best_path
    
    return helper


# Create the three search variants
_minmax = factory()
vct = factory(only_three=True)
vcf = factory(only_four=True)


def minmax(evaluator: BoardEvaluator, role: int, depth: int = 4, enable_vct: bool = True
           ) -> Tuple[float, Optional[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Main minimax search with VCT (Variable-depth Continuous Threat).
    
    Args:
        evaluator: Board evaluator
        role: Current player (1 or -1)
        depth: Search depth (default 4)
        enable_vct: Enable VCT search (default True)
    
    Returns:
        Tuple of (value, move, best_path)
    """
    if enable_vct:
        vct_depth = depth + 8
        
        # First check if we have a winning sequence
        value, move, best_path = vct(evaluator, role, vct_depth)
        if value >= FIVE:
            return value, move, best_path
        
        # Do regular minimax search
        value, move, best_path = _minmax(evaluator, role, depth)
        
        # Check if opponent has a winning sequence after our move
        # If we move and opponent's winning path becomes longer, our move is good
        # If opponent's path stays same or shorter, we should block instead
        evaluator.move(move[0], move[1], role)
        value2, move2, best_path2 = vct(evaluator.reverse(), role, vct_depth)
        evaluator.undo(move[0], move[1])
        
        if value < FIVE and value2 == FIVE and len(best_path2) > len(best_path):
            value3, move3, best_path3 = vct(evaluator.reverse(), role, vct_depth)
            if len(best_path2) <= len(best_path3):
                return value, move2, best_path2  # Use value (not value2) as it's blocked
        
        return value, move, best_path
    else:
        return _minmax(evaluator, role, depth)
