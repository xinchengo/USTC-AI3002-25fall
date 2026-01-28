import os
import random
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import copy
from typing import *
from tqdm import tqdm
import torch

# 可选导入 wandb，如果未安装则跳过
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' to enable experiment tracking.")

# 可选导入 numba，如果未安装则跳过
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not installed. Install with 'pip install numba' to enable JIT acceleration.")

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device is {device}.")


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _count_max_connections_for_single_color_numba(state, board_size, color):
        directions = ((1, 1), (1, 0), (0, 1), (1, -1))
        max_connections = 0
        for i in range(board_size):
            for j in range(board_size):
                for d in range(4):
                    direction_x, direction_y = directions[d]
                    current_pos_x, current_pos_y = i, j
                    current_connections = 0
                    while 0 <= current_pos_x < board_size and 0 <= current_pos_y < board_size:
                        if state[current_pos_x][current_pos_y] == color:
                            current_connections += 1
                        else:
                            break
                        current_pos_x += direction_x
                        current_pos_y += direction_y
                    if current_connections > max_connections:
                        max_connections = current_connections
        return max_connections

    @njit(cache=True)
    def _board_potential_numba(board, board_size, player, bound, base_exp):
        weights_by_openness = (0.0, 0.4, 1.0)
        score = 0.0

        def scan_line(sr, sc, dr, dc):
            line_score = 0.0
            r, c = sr, sc
            run_len = 0
            run_start_r = 0
            run_start_c = 0
            while 0 <= r < board_size and 0 <= c < board_size:
                v = board[r, c]
                if v == player:
                    if run_len == 0:
                        run_start_r = r
                        run_start_c = c
                    run_len += 1
                else:
                    if run_len != 0:
                        head_r = run_start_r - dr
                        head_c = run_start_c - dc
                        head_open = 1 if (0 <= head_r < board_size and 0 <= head_c < board_size and board[head_r, head_c] == 0) else 0
                        tail_open = 1 if v == 0 else 0
                        openness = head_open + tail_open
                        line_score += weights_by_openness[openness] * (float(run_len) / bound) ** base_exp
                        run_len = 0
                r += dr
                c += dc

            if run_len != 0:
                head_r = run_start_r - dr
                head_c = run_start_c - dc
                head_open = 1 if (0 <= head_r < board_size and 0 <= head_c < board_size and board[head_r, head_c] == 0) else 0
                openness = head_open
                line_score += weights_by_openness[openness] * (float(run_len) / bound) ** base_exp
            return line_score

        # Rows (0, 1)
        for i in range(board_size):
            score += scan_line(i, 0, 0, 1)

        # Columns (1, 0)
        for j in range(board_size):
            score += scan_line(0, j, 1, 0)

        # Primary diagonals (1, 1)
        for j in range(board_size):
            score += scan_line(0, j, 1, 1)
        for i in range(1, board_size):
            score += scan_line(i, 0, 1, 1)

        # Secondary diagonals (1, -1)
        for j in range(board_size):
            score += scan_line(0, j, 1, -1)
        for i in range(1, board_size):
            score += scan_line(i, board_size - 1, 1, -1)

        return score


class UtilGobang:
    # Heuristic scoring constants (used by evaluate_board and heuristic reward)
    SCORE_BASE = 10
    SCORE_FIVE = SCORE_BASE ** 5
    SCORE_LIVE_FOUR = SCORE_BASE ** 4
    SCORE_DEAD_FOUR = SCORE_BASE ** 3
    SCORE_LIVE_THREE = SCORE_BASE ** 3
    SCORE_DEAD_THREE = SCORE_BASE ** 2
    SCORE_LIVE_TWO = SCORE_BASE ** 2
    SCORE_DEAD_TWO = SCORE_BASE ** 1
    WIN_REWARD = 10.0

    BASE_GROWTH_EXP = 2.5

    def __init__(self, board_size, bound):
        self.board_size, self.bound = board_size, bound
        self.board = np.zeros((board_size, board_size))
        self.window, self.canvas, self.cell_size = None, None, None
        self.action_space = [(i, j) for i in range(board_size) for j in range(board_size)]
        self.model, self.opponent = None, None

    def restart(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.action_space = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]

    def draw_board(self, random_response, model, opponent):
        opponent_name = "random noise" if random_response else "training model itself"
        print(f"Playing process is being visualized with opponent {opponent_name}.")
        self.model, self.opponent = model, opponent
        self.window = tk.Tk()
        self.window.title("Gobang Board")
        self.canvas = tk.Canvas(self.window, width=400, height=400)
        self.canvas.pack()
        self.cell_size = 400 // self.board_size
        self.visualize_board(random_response)
        self.window.mainloop()

    def visualize_board(self, random_response):
        self.canvas.delete("all")
        color, end_up_gaming = self.update_board(random_response=random_response, learning=False)
        text = "Black wins." if color == 1 else "White wins." if color == 2 else "Tie." if color == 0 else None
        if text is not None:
            message = tk.Message(self.window, text=text, width=100)
            message.pack()
        for i in range(self.board_size):
            for j in range(self.board_size):
                x1 = i * self.cell_size
                y1 = j * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                if self.board[i][j] == 1:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="black")
                elif self.board[i][j] == 2:
                    self.canvas.create_oval(x1, y1, x2, y2, fill="white")
        if end_up_gaming is True:
            print("Game ended.")
        else:
            self.window.after(1000, lambda: self.visualize_board(random_response))

    def judge_legal_position(self, x, y) -> bool:
        return 0 <= x < self.board_size and 0 <= y < self.board_size

    def count_max_connections_for_single_color(self, state, color) -> int:
        if NUMBA_AVAILABLE:
            return int(_count_max_connections_for_single_color_numba(state, self.board_size, color))
        directions = [(1, 1), (1, 0), (0, 1), (1, -1)]
        max_connections = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                for direction_x, direction_y in directions:
                    current_pos_x, current_pos_y = i, j
                    current_connections = 0
                    while self.judge_legal_position(current_pos_x, current_pos_y):
                        if state[current_pos_x][current_pos_y] == color:
                            current_connections += 1
                        else:
                            break
                        current_pos_x += direction_x
                        current_pos_y += direction_y
                    max_connections = max(current_connections, max_connections)
        return max_connections

    def count_max_connections(self, state) -> Tuple[int, int]:
        return (self.count_max_connections_for_single_color(state, 1),
                self.count_max_connections_for_single_color(state, 2))

    @staticmethod
    def array_to_hashable(array):
        return tuple([tuple(r) for r in array])

    @staticmethod
    def hashable_to_array(hash_key):
        return np.array([list(r) for r in hash_key])

    def position_to_index(self, x: int, y: int) -> int:
        return x * self.board_size + y

    def index_to_position(self, index: int) -> Tuple[int, int]:
        x = index // self.board_size
        y = index - x * self.board_size
        return x, y

    @staticmethod
    def identity_transform(state: np.array):
        return np.array([
            [1 if r == 2 else 2 if r == 1 else 0 for r in row] for row in state
        ])

    def sample_action_and_response(self, random_response):
        raise NotImplementedError("Not Implemented!")

    def get_connection_and_reward(self, action, response):
        raise NotImplementedError("Not Implemented!")

    def get_next_state(self, action, response):
        raise NotImplementedError("Not Implemented!")

    def update_board(self, random_response, learning: bool = True, attempt: int = 8) -> Tuple[int, bool]:
        action_space = copy.deepcopy(self.action_space)
        (next_state_free_of_response, next_state,
         current_black_connection, current_white_connection,
         next_black_connection, next_white_connection, reward) = [None, None, None, None, None, None, None]
        for _ in range(attempt if learning else 1):
            self.action_space = copy.deepcopy(action_space)
            action, response = self.sample_action_and_response(random_response)
            (current_black_connection, current_white_connection,
             next_black_connection, next_white_connection, reward) = self.get_connection_and_reward(action, response)
            next_state = self.get_next_state(action, response)
            next_state_free_of_response = self.get_next_state(action, None)
        self.board = next_state_free_of_response if next_black_connection >= self.bound else next_state
        return ((1, True) if next_black_connection >= self.bound else
                (2, True) if next_white_connection >= self.bound else
                (0, True) if len(self.action_space) == 0 else
                (-1, False))

    def evaluate_agent_performance(self, random_response, model, opponent, episodes=1000):

        opponent_name = "random noise" if random_response else "training model itself"
        print(f"Start evaluating with opponent {opponent_name}.")
        self.model, self.opponent = model, opponent
        black_wins, white_wins, ties = 0, 0, 0
        for _ in tqdm(range(episodes)):
            self.restart()
            while True:
                color, end_up_gaming = self.update_board(learning=False, random_response=random_response)
                black_wins, white_wins, ties = ((black_wins, white_wins, ties) if end_up_gaming is False else
                                                (black_wins, white_wins, ties + 1) if color == 0 else
                                                (black_wins + 1, white_wins, ties) if color == 1 else
                                                (black_wins, white_wins + 1, ties))
                if end_up_gaming:
                    # print(f"Black wins: {black_wins}, white wins: {white_wins}, and ties: {ties}.")
                    # print(
                    #     f"The evaluated winning probability for the black pieces is "
                    #     f"{black_wins / (black_wins + white_wins + ties)}."
                    # )
                    break
        self.restart()
        print(f"Evaluation finished. Black wins: {black_wins}, white wins: {white_wins}, and ties: {ties}.")
        print(
            f"The evaluated winning probability for the black pieces is "
            f"{black_wins / (black_wins + white_wins + ties)}."
        )

    def board_potential(self, board, player: int) -> float:
        """
        评估函数：一个棋盘关于某一方的“势能”。
        """

        assert self.bound == 5, "This function only supports standard Gobang with bound 5."
        n = self.board_size
        BOUND = 5
        base_exp = float(self.BASE_GROWTH_EXP)

        board = np.asarray(board, dtype=np.int8)

        if NUMBA_AVAILABLE:
            return float(_board_potential_numba(board, n, player, BOUND, base_exp))

        # openness 指的是连续段的开放端数量，取值范围为 {0, 1, 2}
        weights_by_openness = (0.0, 0.4, 1.0)

        # 扫描线算法
        def scan_line(sr: int, sc: int, dr: int, dc: int) -> float:
            line_score = 0.0
            r, c = sr, sc
            run_len = 0
            run_start_r = 0
            run_start_c = 0
            while 0 <= r < n and 0 <= c < n:
                v = board[r, c]
                if v == player:
                    if run_len == 0:
                        run_start_r = r
                        run_start_c = c
                    run_len += 1
                else:
                    if run_len != 0:
                        head_r = run_start_r - dr
                        head_c = run_start_c - dc
                        head_open = 1 if (0 <= head_r < n and 0 <= head_c < n and board[head_r, head_c] == 0) else 0
                        tail_open = 1 if v == 0 else 0
                        openness = head_open + tail_open
                        line_score += weights_by_openness[openness] * (float(run_len) / BOUND) ** base_exp
                        run_len = 0
                r += dr
                c += dc

            if run_len != 0:
                head_r = run_start_r - dr
                head_c = run_start_c - dc
                head_open = 1 if (0 <= head_r < n and 0 <= head_c < n and board[head_r, head_c] == 0) else 0
                openness = head_open  # tail is out-of-bounds -> blocked
                line_score += weights_by_openness[openness] * (float(run_len) / BOUND) ** base_exp
            return line_score

        score = 0.0

        # Rows (0, 1)
        for i in range(n):
            score += scan_line(i, 0, 0, 1)

        # Columns (1, 0)
        for j in range(n):
            score += scan_line(0, j, 1, 0)

        # Primary diagonals (1, 1)
        for j in range(n):
            score += scan_line(0, j, 1, 1)
        for i in range(1, n):
            score += scan_line(i, 0, 1, 1)

        # Secondary diagonals (1, -1)
        for j in range(n):
            score += scan_line(0, j, 1, -1)
        for i in range(1, n):
            score += scan_line(i, n - 1, 1, -1)

        return float(score)
               
    def evaluate_board(self, board, color: int) -> int:
        """
        评估函数：基于棋型给予分数，而非简单的连珠长度。
        参考了传统五子棋 AI 的启发式评分。

        Notes:
        - Returns a (potentially large) heuristic score for `color` only.
        - This function is used inside reward shaping; reward should be normalized
          separately to avoid exploding critic targets.
        """

        # raise DeprecationWarning("This function is deprecated. Use board_potential instead.")

        board = np.asarray(board, dtype=np.int8)
        score = 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        visited = set()

        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r][c] != color:
                    continue
                
                for dr, dc in directions:
                    if (r, c, dr, dc) in visited:
                        continue
                    
                    prev_r, prev_c = r - dr, c - dc
                    if self.judge_legal_position(prev_r, prev_c) and board[prev_r][prev_c] == color:
                        continue
                    
                    count = 0
                    curr_r, curr_c = r, c
                    while self.judge_legal_position(curr_r, curr_c) and board[curr_r][curr_c] == color:
                        visited.add((curr_r, curr_c, dr, dc))
                        count += 1
                        curr_r += dr
                        curr_c += dc
                    
                    blocked_head = False
                    if not self.judge_legal_position(prev_r, prev_c) or board[prev_r][prev_c] != 0:
                        blocked_head = True
                        
                    blocked_tail = False
                    if not self.judge_legal_position(curr_r, curr_c) or board[curr_r][curr_c] != 0:
                        blocked_tail = True
                    
                    if count >= 5:
                        score += self.SCORE_FIVE
                    elif count == 4:
                        if not blocked_head and not blocked_tail: # 活四
                            score += self.SCORE_LIVE_FOUR
                        elif not (blocked_head and blocked_tail): # 冲四 (死四)
                            score += self.SCORE_DEAD_FOUR
                    elif count == 3:
                        if not blocked_head and not blocked_tail: # 活三
                            score += self.SCORE_LIVE_THREE
                        elif not (blocked_head and blocked_tail): # 眠三
                            score += self.SCORE_DEAD_THREE
                    elif count == 2:
                        if not blocked_head and not blocked_tail: # 活二
                            score += self.SCORE_LIVE_TWO
                        elif not (blocked_head and blocked_tail): # 眠二
                            score += self.SCORE_DEAD_TWO
        return score


class Gobang(UtilGobang):

    def __init__(self, board_size, bound, training, reward_type='default'):
        super().__init__(board_size=board_size, bound=bound)
        self.training = training
        self.reward_type = reward_type
        self.model, self.opponent = None, None

    def get_next_state(self, action: Tuple[int, int, int], response: Tuple[int, int, int]) -> np.array:
        black, xb, yb = action
        next_state = copy.deepcopy(self.board)
        next_state[xb][yb] = black

        if response is not None:
            white, x_white, y_white = response
            next_state[x_white][y_white] = white
        return next_state

    def sample_response(self, random_response, x, y) -> Union[Tuple[int, int, int], None]:
        if self.action_space:
            state = self.identity_transform(self.board)
            state[x][y] = 2
            policy = self.opponent.actor(state)[0].detach().cpu().numpy()
            if random_response:
                policy = [1 if p > 0 else 0 for p in policy]
                policy = [p / sum(policy) for p in policy]
            n = state.shape[0]
            action = np.random.choice(range(self.board_size ** 2), p=policy)
            x_, y_ = _index_to_position(n, action)
            self.action_space.remove((x_, y_))
            return 2, x_, y_
        else:
            return None

    def get_connection_and_reward(self, action: Tuple[int, int, int],
                                  response: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int, int, float]:
        
        
        # Calculate reward based on reward_type
        if self.reward_type == 'default':
            next_state = self.get_next_state(action, response)
            black_1, white_1 = self.count_max_connections(self.board)
            black_2, white_2 = self.count_max_connections(next_state)
            reward = (black_2 ** 2 - white_2 ** 2) - (black_1 ** 2 - white_1 ** 2)
            return black_1, white_1, black_2, white_2, reward
        
        elif self.reward_type == 'sparse':
            # Sparse terminal-only reward.
            # If black wins immediately after its move, opponent should NOT place a response.
            # If white wins after the response, it's a loss for black.
            # Otherwise reward is 0.
            black_1, white_1 = self.count_max_connections(self.board)

            next_state_after_black = self.get_next_state(action, None)
            black_2_black_only, white_2_black_only = self.count_max_connections(next_state_after_black)
            if black_2_black_only >= self.bound:
                return black_1, white_1, black_2_black_only, white_1, float(self.WIN_REWARD)

            next_state = self.get_next_state(action, response)
            black_2, white_2 = self.count_max_connections(next_state)
            if white_2 >= self.bound:
                return black_1, white_1, black_2, white_2, float(-self.WIN_REWARD)

            return black_1, white_1, black_2, white_2, 0.0
        
        elif self.reward_type == 'potential':
            # PPO-friendly potential-based reward shaping.
            defensive_awareness = 0.9
            b1, w1 = self.count_max_connections(self.board)

            # Potential definition: encourage our growth while discouraging opponent growth.
            phi_1 = self.board_potential(self.board, 1) - defensive_awareness * self.board_potential(self.board, 2)

            # Terminal handling must match game dynamics: if black wins after its move,
            # the opponent should NOT place a response.
            next_state_after_black = self.get_next_state(action, None)
            b2_black_only, w2_black_only = self.count_max_connections(next_state_after_black)
            if b2_black_only >= self.bound:
                return b1, w1, b2_black_only, w2_black_only, float(self.WIN_REWARD)

            next_state = self.get_next_state(action, response)
            b2, w2 = self.count_max_connections(next_state)
            phi_2 = self.board_potential(next_state, 1) - defensive_awareness * self.board_potential(next_state, 2)
            reward = float(phi_2 - phi_1)

            # If white wins after the response, treat as terminal loss.
            if w2 >= self.bound:
                reward = float(-self.WIN_REWARD)
            return b1, w1, b2, w2, reward
        
        
        elif self.reward_type == 'heuristic':
            score_black_1 = self.evaluate_board(self.board, 1)
            score_white_1 = self.evaluate_board(self.board, 2)

            b1, w1 = self.count_max_connections(self.board)

            # Evaluate after black move first (terminal handling must ignore response).
            next_state_after_black = self.get_next_state(action, None)
            score_black_2_black_only = self.evaluate_board(next_state_after_black, 1)
            score_white_2_black_only = self.evaluate_board(next_state_after_black, 2)
            b2_black_only, w2_black_only = self.count_max_connections(next_state_after_black)

            next_state = self.get_next_state(action, response)
            score_black_2 = self.evaluate_board(next_state, 1)
            score_white_2 = self.evaluate_board(next_state, 2)
            b2, w2 = self.count_max_connections(next_state)

            # 使用分数差作为奖励的基础
            value_1 = score_black_1 - score_white_1
            value_2 = score_black_2 - score_white_2
            delta = value_2 - value_1

            # 使用 tanh 进行归一化，避免奖励过大
            scale = float(self.SCORE_LIVE_FOUR) # 此尺度按照经验得到
            reward = float(np.tanh(delta / (scale + 1e-6)))

            # 终局处理
            # 如果黑棋获胜，则忽略之前的奖励，直接给出固定胜利奖励
            if b2_black_only >= self.bound or score_black_2_black_only >= self.SCORE_FIVE:
                reward = self.WIN_REWARD
                return b1, w1, b2_black_only, w2_black_only, reward
            elif score_white_2 >= self.SCORE_FIVE or w2 >= self.bound:
                reward = -self.WIN_REWARD

            # # Reward    
            # scale = 100.0
            
            # diff_black = (score_black_2 - score_black_1) / scale
            # diff_white = (score_white_2 - score_white_1) / scale
            
            # reward = diff_black - 0.8 * diff_white
            
            # # 如果这一步直接赢了，给予巨大额外奖励
            # if score_black_2 >= 100000: 
            #     reward += 100.0

            # b1, w1 = self.count_max_connections(self.board)


            # # 使用分数差作为奖励的基础
            # value_1 = score_black_1 - score_white_1
            # value_2 = score_black_2 - score_white_2
            # delta = value_2 - value_1

            # # 使用 tanh 进行归一化，避免奖励过大
            # scale = float(self.SCORE_LIVE_FOUR) # 此尺度按照经验得到
            # reward = float(np.tanh(delta / (scale + 1e-6)))

            # # 终局处理
            # # 如果黑棋获胜，则忽略之前的奖励，直接给出固定胜利奖励
            # if b2_black_only >= self.bound or score_black_2_black_only >= self.SCORE_FIVE:
            #     reward = self.WIN_REWARD
            #     return b1, w1, b2_black_only, w2_black_only, reward
            # elif score_white_2 >= self.SCORE_FIVE or w2 >= self.bound:
            #     reward = -self.WIN_REWARD

            return b1, w1, b2, w2, reward
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def sample_action_and_response(self, random_response) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        state = self.board
        policy = self.model.actor(state)[0].detach().cpu().numpy()
        n = state.shape[0]
        action = np.random.choice(range(self.board_size ** 2), p=policy)
        x, y = _index_to_position(n, action)
        self.action_space.remove((x, y))
        return (1, x, y), self.sample_response(random_response, x, y)


def _position_to_index(board_size, x: int, y: int) -> int:
    return int(x * board_size + y)


def _index_to_position(board_size, index: int) -> Tuple[int, int]:
    x = index // board_size
    y = index - x * board_size
    return x, y


def _sample_response(chessboard, actor, x, y):
    state = chessboard.identity_transform(chessboard.board)
    state[x][y] = 2
    policy = actor(state)[0].detach().cpu().numpy()
    n = state.shape[0]
    action = np.random.choice(range(chessboard.board_size ** 2), p=policy)
    x_, y_ = _index_to_position(n, action)
    chessboard.action_space.remove((x_, y_))
    return 2, x_, y_


def track_loss(actor_records, critic_records, entropy):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 20))

    ax1.plot(actor_records, label='Actor Loss', color='green')
    ax1.set_title('Actor Loss Tracking')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(critic_records, label='Critic Loss', color='red')
    ax2.set_title('Critic Loss Tracking')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    ax3.plot(entropy, label='Policy Entropy', color='blue')
    ax3.set_title('Policy Entropy Tracking')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Entropy')
    ax3.legend()
    ax3.grid(True)
    ax3.figure.savefig("loss_tracker.png")
    plt.close()


def _sample_action_and_response(chessboard, actor, state):
    policy = actor(state)[0].detach().cpu().numpy()
    n = state.shape[0]
    action = np.random.choice(range(actor.board_size ** 2), p=policy)
    x, y = _index_to_position(n, action)
    response = None if len(np.nonzero(state == 0)[0]) <= 1 else _sample_response(chessboard, actor, x, y)
    return (1, x, y), response


def _get_next_state(state, action, response):
    black, xb, yb = action
    next_state = copy.deepcopy(state)
    next_state[xb][yb] = black
    if response is not None:
        white, x_white, y_white = response
        next_state[x_white][y_white] = white
    return next_state


def train_model(model, num_episodes=1000, checkpoint=1000, gamma=0.5, save_dir="checkpoints", reward_type='default'):
    chess_board = Gobang(board_size=model.board_size, bound=model.bound, training=True, reward_type=reward_type)
    actor_records, critic_records, entropy_records = [], [], []
    for _ in range(num_episodes):
        states, actions, rewards, next_states = [[] for _ in range(4)]
        chess_board.restart()
        for count in range(chess_board.board_size ** 2 // 2 + 1):
            state = copy.deepcopy(chess_board.board)
            action, response = _sample_action_and_response(chess_board, model.actor, state)
            next_state = _get_next_state(state, action, response)
            black_1, white_1, black_2, white_2, reward = chess_board.get_connection_and_reward(action=action,
                                                                                               response=response)

            stop = True if (black_2 >= model.bound or white_2 >= model.bound
                            or len(np.nonzero(next_state == 0)[0]) == 0) else False

            # Keep terminal-reward handling consistent with the selected reward_type.
            # The original code overwrote heuristic rewards with the default formula.
            if black_2 >= model.bound:
                # If black wins immediately, there is no opponent response in the environment transition.
                next_state = _get_next_state(state, action, None)
                if reward_type == 'default':
                    white_2 = white_1
                    reward = (black_2 ** 2 - white_2 ** 2) - (black_1 ** 2 - white_1 ** 2)

            states.append([state])
            actions.append([action[1], action[2]])
            rewards.append(reward)
            chess_board.board = next_state
            if stop:
                break

        states = torch.tensor(np.array(states)).to(torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards)).to(torch.float32).to(device)
        actions = torch.tensor(np.array(actions)).to(torch.float32).to(device)

        policy, qs = model(states, actions)
        next_qs = qs[1:]
        next_qs = torch.cat((next_qs, torch.tensor([0]).to(device)))

        entropy = -float(torch.mean(torch.sum(policy * torch.log(policy + 1e-6), dim=1)).detach())
        entropy_records.append(entropy)

        actor_loss, critic_loss = model.optimize(policy, qs, actions, rewards, next_qs, gamma)
        actor_records.append(float(actor_loss))
        critic_records.append(float(critic_loss))
        
        if WANDB_AVAILABLE:
            wandb.log({
                "episode": _,
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "entropy": entropy,
                "actor_loss_neg": -float(actor_loss),
            })
        
        print(
            f"Episode {_} / {num_episodes}: Actor Loss {-actor_loss}, Critic Loss "
            f"{critic_loss}.")
        if (_ + 1) % 10 == 0:
            try:
                track_loss(actor_records, critic_records, entropy_records)
            except Exception as e:
                print(e)
        if (_ + 1) % checkpoint == 0:
            # 使用传入的 save_dir
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/model_{_}.pth")

            # Also save the complete model object as pickle
            import pickle
            with open(f"{save_dir}/model_{_}.pkl", 'wb') as f:
                pickle.dump(model, f)


__all__ = ['_position_to_index', '_index_to_position', '_sample_response', 'train_model',
           '_sample_action_and_response', '_get_next_state', 'UtilGobang', 'Gobang', 'device']
