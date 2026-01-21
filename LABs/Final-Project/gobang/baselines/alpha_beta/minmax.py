"""
Minimax with Alpha-Beta pruning for Gobang.
Ported from https://github.com/lihongxun945/gobang
"""

from .cache import Cache
from .evaluate import FIVE

MAX = 1000000000
ONLY_THREE_THRESHOLD = 6

# Shared cache for minmax, vct, and vcf
_cache = Cache()


# Cache hit statistics
cache_hits = {
    'search': 0,
    'total': 0,
    'hit': 0
}


def _factory(only_three=False, only_four=False):
    """Factory to create minimax helper functions."""
    
    def helper(board, role, depth, c_depth=0, path=None, alpha=-MAX, beta=MAX):
        """
        Minimax with alpha-beta pruning.
        
        Args:
            board: Board instance
            role: Current role (1 for black, -1 for white)
            depth: Total search depth
            c_depth: Current search depth
            path: Current path
            alpha: Alpha value for pruning
            beta: Beta value for pruning
        
        Returns:
            (value, move, path)
        """
        if path is None:
            path = []
        
        cache_hits['search'] += 1
        
        if c_depth >= depth or board.is_game_over():
            return (board.evaluate(role), None, list(path))
        
        hash_val = board.hash()
        prev = _cache.get(hash_val)
        if prev and prev.get('role') == role:
            if ((abs(prev['value']) >= FIVE or prev['depth'] >= depth - c_depth) and
                prev.get('only_three') == only_three and prev.get('only_four') == only_four):
                cache_hits['hit'] += 1
                return (prev['value'], prev['move'], list(path) + prev['path'])
        
        value = -MAX
        move = None
        best_path = list(path)
        best_depth = 0
        
        points = board.get_valuable_moves(
            role, c_depth,
            only_three or c_depth > ONLY_THREE_THRESHOLD,
            only_four
        )
        
        if not points:
            return (board.evaluate(role), None, list(path))
        
        # Iterative deepening
        for d in range(c_depth + 1, depth + 1):
            # Only search even depths for finding wins
            if d % 2 != 0:
                continue
            
            break_all = False
            for point in points:
                board.put(point[0], point[1], role)
                new_path = list(path) + [point]
                
                current_value, current_move, current_path = helper(
                    board, -role, d, c_depth + 1, new_path, -beta, -alpha
                )
                current_value = -current_value
                
                board.undo()
                
                # During iterative deepening, only consider winning moves
                if current_value >= FIVE or d == depth:
                    # Even for losing moves, choose the longest path
                    if (current_value > value or
                        (current_value <= -FIVE and value <= -FIVE and len(current_path) > best_depth)):
                        value = current_value
                        move = point
                        best_path = current_path
                        best_depth = len(current_path)
                
                alpha = max(alpha, value)
                if alpha >= FIVE:  # Found a win, stop searching
                    break_all = True
                    break
                if alpha >= beta:  # Beta cutoff
                    break
            
            if break_all:
                break
        
        # Cache the result
        if ((c_depth < ONLY_THREE_THRESHOLD or only_three or only_four) and
            (not prev or prev['depth'] < depth - c_depth)):
            cache_hits['total'] += 1
            _cache.put(hash_val, {
                'depth': depth - c_depth,
                'value': value,
                'move': move,
                'role': role,
                'path': best_path[c_depth:],
                'only_three': only_three,
                'only_four': only_four,
            })
        
        return (value, move, best_path)
    
    return helper


# Create specialized search functions
_minmax = _factory()
vct = _factory(only_three=True)  # Victory by Continuous Threat
vcf = _factory(only_four=True)   # Victory by Continuous Four


def minmax(board, role, depth=4, enable_vct=True):
    """
    Main minimax entry point.
    
    Args:
        board: Board instance
        role: Current role
        depth: Search depth
        enable_vct: Whether to enable VCT search
    
    Returns:
        (value, move, path)
    """
    if enable_vct:
        vct_depth = depth + 8
        
        # First check if we have a winning sequence
        value, move, best_path = vct(board, role, vct_depth)
        if value >= FIVE:
            return (value, move, best_path)
        
        # Normal minimax search
        value, move, best_path = _minmax(board, role, depth)
        
        if move is None:
            return (value, move, best_path)
        
        # Check opponent's winning threat after our move
        board.put(move[0], move[1], role)
        value2, move2, best_path2 = vct(board.reverse(), role, vct_depth)
        board.undo()
        
        if value < FIVE and value2 == FIVE and len(best_path2) > len(best_path):
            # Check opponent's threat without our move
            value3, move3, best_path3 = vct(board.reverse(), role, vct_depth)
            if len(best_path2) <= len(best_path3):
                return (value, move2, best_path2)
        
        return (value, move, best_path)
    else:
        return _minmax(board, role, depth)


def reset_cache():
    """Reset the search cache."""
    global _cache
    _cache = Cache()
    cache_hits['search'] = 0
    cache_hits['total'] = 0
    cache_hits['hit'] = 0
