from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--num_episodes', type=int, help='number of episodes')
parser.add_argument('--checkpoint', type=int, help='the interval of saving models')
parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
parser.add_argument('--use_deep', action='store_true', help='use deep cnn architecture') # 新增定义 use_deep

args = parser.parse_args()
num_episodes = args.num_episodes
checkpoint = args.checkpoint


class Actor(nn.Module):
    """
    The actor is responsible for generating dependable policies to maximize the cumulative reward as much as possible.
    It takes a batch of arrays shaped either (B, 1, N, N) or (N, N) as input, and outputs a tensor shaped (B, N ** 2)
    as the generated policy.
    """

    def __init__(self, board_size: int, lr=1e-4, use_deep=False):
        super().__init__()
        self.board_size = board_size
        self.use_deep = use_deep  # 标记是否使用深层网络
        """
        # Define your NN structures here. Torch modules have to be registered during the initialization process.
        # For example, you can define CNN structures as follows:

        # self.conv_blocks = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=kernel_size, padding=padding),
        #     nn.MaxPool2d(kernel_size=kernel_size, padding=padding, stride=stride),
        #     nn.ReLU(),
        # )

        # Here, channels, kernel_size, padding, and stride are what we would call "Hyperparameters" in deep learning.

        # After convolution, you can flatten (nn.Flatten()) the hidden 2d-representation to obtain the corresponding
        # 1d-representation. Then, fully connected layers can be used to obtain a representation of n**2 dimensions,
        # with each digit indicating the "raw number of policy" (which has to be further constrained and modified
        # in the next step).

        # self.linear_blocks = nn.Sequential(
        #     nn.Linear(in_features=features, out_features=board_size ** 2),
        # )

        # After obtaining a representation of n**2 dimensions, you STILL NEED TO PERFORM ADDITIONAL PROCESSING,
        # including:
        # i) ensuring that all digits corresponding to illegal actions are set to 0 (!!!!!THE MOST IMPORTANT!!!!!);
        # ii) ensuring that the remaining digits satisfy the normalization condition (i.e., the sum of them is equal
        #     to 1).
        # In-place operations are strongly discouraged because they can lead to gradient calculation failures.
        # As an intelligent alternative, consider approaches that can avoid in-place modifications to achieve the goal.

        # You are also encouraged to explore other powerful models and experiment with different techniques,
        # such as using attention modules, different activation functions, or simply adjusting hyperparameter settings.
        """

        # BEGIN YOUR CODE
        if not self.use_deep:
            # Architecture 1: Baseline CNN (3-Layer)
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            flat_features = 128 * board_size * board_size
        else:
            # Architecture 2: Deep CNN (5-Layer)
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.relu4 = nn.ReLU()
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.relu5 = nn.ReLU()
            flat_features = 256 * board_size * board_size
        
        self.fc = nn.Linear(flat_features, board_size * board_size)
        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray):
        if len(x.shape) == 2:
            output = torch.tensor(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.tensor(x).to(device).to(torch.float32)

        # Further process and transform the data here. Ensure that the output is shaped (B, n ** 2).
        # We have already ensured that the shape of the raw input is unified to be (B, 1, N, N),
        # where B >= 1 represents the number of data in this batch, and N = n is exactly the size of the board.

        # You can continue processing the data here using the modules that were previously registered during the
        # initialization process. For example:

        # output = self.conv_blocks(output)
        # output = nn.Flatten()(output)
        # output = self.linear_blocks(output)

        # And the reminder AGAIN:

        # ****************************************
        # After obtaining a representation of n**2 dimensions, you STILL NEED TO PERFORM ADDITIONAL DATA PROCESSING,
        # including:
        # i) ensuring that all digits corresponding to illegal actions are set to 0 (!!!!!THE MOST IMPORTANT!!!!!);
        # ii) ensuring that the remaining digits satisfy the normalization condition (i.e., the sum of them is equal
        #     to 1).
        # In-place operations are strongly discouraged because they can lead to gradient calculation failures.
        # ****************************************

        # BEGIN YOUR CODE
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(device).float()
        else:
            x = x.to(device).float()
        if x.dim() == 2: output = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3: output = x.unsqueeze(0)
        else: output = x
        
        # 1. 网络前向传播：通过卷积层提取特征
        if not self.use_deep:
            out = self.relu1(self.conv1(output))
            out = self.relu2(self.conv2(out))
            out = self.relu3(self.conv3(out))
        else:
            out = self.relu1(self.conv1(output))
            out = self.relu2(self.conv2(out))
            out = self.relu3(self.conv3(out))
            out = self.relu4(self.conv4(out))
            out = self.relu5(self.conv5(out))
        
        # 2. 展平张量：从 (B, Channels, N, N) 变为 (B, Channels*N*N)
        out = out.view(out.size(0), -1)
        
        # 3. 全连接层：得到每个位置的原始评分
        # 形状变为 (B, N^2)
        logits = self.fc(out)

        # 4. 合法动作掩码
        # 输入 output 是 (B, 1, N, N)，展平对应 logits 的 (B, N^2)
        flat_board = output.view(output.size(0), -1)
        
        # 创建掩码
        illegal_mask = (flat_board != 0)
        # 注：使用 logits[mask] = -inf 会破坏梯度计算
        logits = logits.masked_fill(illegal_mask, -1e9)

        # 5. 归一化
        output = torch.softmax(logits, dim=1)
        # END YOUR CODE
        return output


class Critic(nn.Module):
    """
    The critic is responsible for generating dependable Q-values to fit the solution of Bellman Equations. It takes
    a batch of arrays (shaped either (B, 1, N, N) or (N, N)) and a batch of actions (shaped (B, 2)) as input, and
    outputs a tensor shaped (B, ) as the Q-values on the specified (s, a) pairs.

    For example, actions can be:
    [[0, 1],
     [2, 3],
     [5, 6]]
    which means that there are three actions leading the model to place the pieces on the coordinates (0, 1), (2, 3),
    and (5, 6), respectively. These actions correspond one-to-one with indices 0 * 12 + 1 = 1, 2 * 12 + 3 = 27,
    and 5 * 12 + 6 = 66, assuming n to be 12. You can easily transform a single action to the corresponding digit by
    using _position_to_index, or using _index_to_position vice versa.

    The main idea is that we first obtain a tensor shaped (B, N ** 2) as the Q-values for all possible actions given
    the unified state tensor shaped (B, 1, N, N), and then extract the Q-values corresponding to each action (i, j)
    from the entire Q-value tensor. (_position_to_index should be fully utilized to get the corresponding action indices).
    Finally, it returns a tensor of shape (B,) containing these Q-values.
    """

    def __init__(self, board_size: int, lr=1e-4):
        super().__init__()
        self.board_size = board_size
        # Define your NN structures here as the same. Torch modules have to be registered during the initialization
        # process.

        # BEGIN YOUR CODE
        # 神经网络结构
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        # 为棋盘上的每个位置生成一个Q值
        self.fc = nn.Linear(128 * board_size * board_size, board_size * board_size)
        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray, action: np.ndarray):
        indices = torch.tensor([_position_to_index(self.board_size, x, y) for x, y in action]).to(device)
        if len(x.shape) == 2:
            output = torch.tensor(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.tensor(x).to(device).to(torch.float32)

        # BEGIN YOUR CODE
        # 前向传播
        out = self.relu1(self.conv1(output))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))
        
        out = out.view(out.size(0), -1)
        
        # B，N^2
        q_values_all = self.fc(out)
        
        # 提取指定动作的 Q 值
        output = q_values_all.gather(1, indices.unsqueeze(1))
        # 必须 squeeze 掉最后一维，否则会导致 utils.py 中的维度不匹配错误
        output = output.squeeze(1)
        # END YOUR CODE

        return output


class GobangModel(nn.Module):
    """
    The GobangModel class integrates the Actor and Critic classes for computation and training. Given state tensors "x"
    and action tensors "action", it directly outputs self.actor(x) and self.critic(x, action) as the policy and Q-values
    respectively.
    """
    # 增加 use_deep 参数
    def __init__(self, board_size: int, bound: int, use_deep=False):
        super().__init__()
        self.bound = bound
        self.board_size = board_size

        """
        Register the actor and critic modules here. You do not need to further design the structures at this step.
        Feel free to add extra parameters in the __init__ method of either the Actor class or the Critic class for your 
        convenience, if necessary.
        """

        # BEGIN YOUR CODE
        # self.actor = Actor(board_size=board_size, ...)
        # self.critic = Critic(board_size=board_size, ...)
        # Register Actor and Critic
        self.actor = Actor(board_size=board_size, use_deep=use_deep)
        self.critic = Critic(board_size=board_size)
        # END YOUR CODE

        self.to(device)

    def forward(self, x, action):
        """
        Return the policy vector π(s) and Q-values Q(s, a) given state "x" and action "action".
        """
        return self.actor(x), self.critic(x, action)

    def optimize(self, policy, qs, actions, rewards, next_qs, gamma, eps=1e-6):
        """
        This function calculates the loss for both the actor and critic.
        Using the obtained loss, we can apply optimization algorithms through actor.optimizer and critic.optimizer
        to either maximize the actor's actual objective or minimize the critic's loss.

        There are 3 bugs in the function "optimize" that prevent the model from executing optimizations correctly.
        Identify and debug all errors.
        """

        # Bug 1：分离 next_qs
        # next_qs 代表目标值。它不应该跟踪梯度。
        targets = rewards + gamma * next_qs.detach()
        
        critic_loss = nn.MSELoss()(targets, qs)
        indices = torch.tensor([_position_to_index(self.board_size, x, y) for x, y in actions]).to(device)
        aimed_policy = policy[torch.arange(len(indices)), indices]
        actor_loss = -torch.mean(torch.log(aimed_policy + eps) * qs.clone().detach())

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        # Bug 2: Actor Optimizer Step
        self.actor.optimizer.step()
        
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        # Bug 3: Critic Optimizer Step
        self.critic.optimizer.step()
        
        return actor_loss, critic_loss


if __name__ == "__main__":
    # 确保 num_episodes 和 checkpoint 有值
    num_episodes = args.num_episodes if args.num_episodes is not None else 1000
    checkpoint = args.checkpoint if args.checkpoint is not None else 500
    
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config={
                    "num_episodes": num_episodes,
                    "checkpoint": checkpoint,
                    "board_size": 12,
                    "bound": 5,
                }
            )
            print("Wandb initialized successfully.")
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb' to enable experiment tracking.")
            print("Continuing without wandb...")
    else:
        # 即使不开启 wandb，也初始化一个 disabled 的 wandb，防止 utils.py 报错
        try:
            import wandb
            wandb.init(mode="disabled")
        except ImportError:
            pass
        
    agent = GobangModel(board_size=12, bound=5, use_deep=args.use_deep).to(device)
    print(f"Model initialized. Use Deep CNN: {args.use_deep}")
    
    # 根据是否使用 deep 决定保存路径
    save_folder = "checkpoints_deep" if args.use_deep else "checkpoints_baseline"
    # 传递 save_dir 给 train_model
    train_model(agent, num_episodes=num_episodes, checkpoint=checkpoint, save_dir=save_folder)
    
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
