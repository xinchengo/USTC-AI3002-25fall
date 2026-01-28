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

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device is {device}.")


class UtilGobang:
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
    
    def evaluate_board(self, board, color):
        """
        评估函数：基于棋型给予分数，而非简单的连珠长度。
        参考了传统五子棋 AI 的启发式评分。
        """
        score = 0
        # 定义棋型分数
        SCORE_FIVE = 100000
        SCORE_LIVE_FOUR = 10000
        SCORE_DEAD_FOUR = 1000
        SCORE_LIVE_THREE = 1000
        SCORE_DEAD_THREE = 100
        SCORE_LIVE_TWO = 100
        SCORE_DEAD_TWO = 10
        
        # 获取所有行、列、对角线
        lines = []
        # 横向
        for r in range(self.board_size):
            lines.append(board[r, :])
        # 纵向
        for c in range(self.board_size):
            lines.append(board[:, c])
        # 对角线
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
                        score += SCORE_FIVE
                    elif count == 4:
                        if not blocked_head and not blocked_tail: # 活四
                            score += SCORE_LIVE_FOUR
                        elif not (blocked_head and blocked_tail): # 冲四 (死四)
                            score += SCORE_DEAD_FOUR
                    elif count == 3:
                        if not blocked_head and not blocked_tail: # 活三
                            score += SCORE_LIVE_THREE
                        elif not (blocked_head and blocked_tail): # 眠三
                            score += SCORE_DEAD_THREE
                    elif count == 2:
                        if not blocked_head and not blocked_tail: # 活二
                            score += SCORE_LIVE_TWO
                        elif not (blocked_head and blocked_tail): # 眠二
                            score += SCORE_DEAD_TWO
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
                                  response: Tuple[int, int, int]) -> Tuple[int, int, int, int, float]:
        
        
        # Calculate reward based on reward_type
        if self.reward_type == 'default':
            next_state = self.get_next_state(action, response)
            black_1, white_1 = self.count_max_connections(self.board)
            black_2, white_2 = self.count_max_connections(next_state)
            reward = (black_2 ** 2 - white_2 ** 2) - (black_1 ** 2 - white_1 ** 2)
            return black_1, white_1, black_2, white_2, reward
        
        elif self.reward_type == 'heuristic':
            score_black_1 = self.evaluate_board(self.board, 1)
            score_white_1 = self.evaluate_board(self.board, 2)
            next_state = self.get_next_state(action, response)
            score_black_2 = self.evaluate_board(next_state, 1)
            score_white_2 = self.evaluate_board(next_state, 2)
            
            # Reward    
            scale = 100.0
            
            diff_black = (score_black_2 - score_black_1) / scale
            diff_white = (score_white_2 - score_white_1) / scale
            
            reward = diff_black - 0.8 * diff_white
            
            # 如果这一步直接赢了，给予巨大额外奖励
            if score_black_2 >= 100000: 
                reward += 100.0

            b1, w1 = self.count_max_connections(self.board)
            b2, w2 = self.count_max_connections(next_state)
            
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

            if black_2 >= model.bound:
                next_state = _get_next_state(state, action, None)
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
