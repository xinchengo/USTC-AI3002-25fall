import os
from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse
import json


class Actor(nn.Module):
    """
    The actor is responsible for generating dependable policies to maximize the cumulative reward as much as possible.
    It takes a batch of arrays shaped either (B, 1, N, N) or (N, N) as input, and outputs a tensor shaped (B, N ** 2)
    as the generated policy.
    """

    def __init__(self, board_size: int, lr=1e-4, model_type: str = "default", extra_specs: dict = None):
        super().__init__()
        self.board_size = board_size
        self.model_type = model_type
        self.extra_specs = extra_specs or {}

        # Default values for backward compatibility
        use_deep = self.extra_specs.get('use_deep', False)

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
        if model_type == "default":
            if not use_deep:
                # Architecture 1: Baseline CNN (3-Layer)
                self.conv_blocks = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU()
                )
                flat_features = 128 * board_size * board_size
            else:
                # Architecture 2: Deep CNN (5-Layer)
                self.conv_blocks = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU()
                )
                flat_features = 256 * board_size * board_size

            self.fc = nn.Linear(flat_features, board_size * board_size)
        elif model_type == "custom":
            # Allow custom architecture based on extra_specs
            channels = self.extra_specs.get('channels', [32, 64, 128])
            kernel_size = self.extra_specs.get('kernel_size', 3)
            padding = self.extra_specs.get('padding', 1)

            conv_layers = []
            in_channels = 1

            for out_channels in channels:
                conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
                conv_layers.append(nn.ReLU())
                in_channels = out_channels

            self.conv_blocks = nn.Sequential(*conv_layers)
            flat_features = channels[-1] * board_size * board_size
            self.fc = nn.Linear(flat_features, board_size * board_size)
        else:
            # Default to baseline architecture if unknown model type
            self.conv_blocks = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )
            flat_features = 128 * board_size * board_size
            self.fc = nn.Linear(flat_features, board_size * board_size)

        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray):
        if isinstance(x, torch.Tensor):
            output = x.detach().clone().to(device).to(torch.float32)
            if len(output.shape) == 2:
                output = output.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            output = torch.from_numpy(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.from_numpy(x).to(device).to(torch.float32)

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
        out = self.conv_blocks(output)
        
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
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 为棋盘上的每个位置生成一个Q值
        self.fc = nn.Linear(128 * board_size * board_size, board_size * board_size)
        # END YOUR CODE

        # Define your optimizer here, which is responsible for calculating the gradients and performing optimizations.
        # The learning rate (lr) is another hyperparameter that needs to be determined in advance.
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x: np.ndarray, action: np.ndarray):
        indices = torch.tensor([_position_to_index(self.board_size, x, y) for x, y in action]).to(device)
        if isinstance(x, torch.Tensor):
            output = x.detach().clone().to(device).to(torch.float32)
            if len(output.shape) == 2:
                output = output.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            output = torch.from_numpy(x).to(device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            output = torch.from_numpy(x).to(device).to(torch.float32)

        # BEGIN YOUR CODE
        # 前向传播
        out = self.conv_blocks(output)
        
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
    def __init__(self, board_size: int, bound: int, model_type: str = "default", extra_specs: dict = None):
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
        self.actor = Actor(board_size=board_size, model_type=model_type, extra_specs=extra_specs)
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
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--num_episodes', type=int, help='number of episodes')
    parser.add_argument('--checkpoint', type=int, help='the interval of saving models')
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for experiment tracking (requires wandb installed)')
    parser.add_argument('--wandb_project', type=str, default='gobang-rl-AI3002', help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
    parser.add_argument('--model-type', type=str, default='default', dest='model_type', help='model architecture type')
    parser.add_argument('--extra-specs', type=str, default='{}', dest='extra_specs', help='extra specifications as JSON string')
    args = parser.parse_args()

    # Parse extra_specs from JSON string
    try:
        extra_specs = json.loads(args.extra_specs)
    except json.JSONDecodeError:
        print(f"Invalid JSON for extra_specs: {args.extra_specs}")
        extra_specs = {}
        # Check if args.extra_specs is a boolean flag (for backward compatibility)
        if args.extra_specs.lower() in ('true', '1', 'yes', 'on'):
            extra_specs = {'use_deep': True}
        elif args.extra_specs.lower() in ('false', '0', 'no', 'off'):
            extra_specs = {'use_deep': False}

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
        
    agent = GobangModel(board_size=12, bound=5, model_type=args.model_type, extra_specs=extra_specs).to(device)
    print(f"Model initialized. Model Type: {args.model_type}, Extra Specs: {extra_specs}")
    # 打印模型参数量
    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # 根据模型类型决定保存路径
    from time import strftime, localtime
    timestamp = strftime("%Y%m%d-%H%M%S", localtime())
    use_deep = extra_specs.get('use_deep', False)
    save_folder = f'checkpoints/{args.model_type}_{"deep_" if use_deep else ""}gobang_model_{timestamp}'
    print(f"Models will be saved to: {save_folder}")
    os.makedirs(save_folder, exist_ok=True)
    # 传递 save_dir 给 train_model
    train_model(agent, num_episodes=num_episodes, checkpoint=checkpoint, save_dir=save_folder)

    import pickle
    # 保存 最终模型
    final_model_path = os.path.join(save_folder, 'final_model.pth')
    torch.save(agent.state_dict(), final_model_path)

    # 保存完整模型对象为pickle文件
    final_pickle_path = os.path.join(save_folder, 'final_model.pkl')
    with open(final_pickle_path, 'wb') as f:
        pickle.dump(agent, f)

    # 保存超参数
    hyperparams_path = os.path.join(save_folder, 'hyperparameters.txt')
    with open(hyperparams_path, 'w') as f:
        f.write(f'num_episodes: {num_episodes}\n')
        f.write(f'checkpoint: {checkpoint}\n')
        f.write(f'model_type: {args.model_type}\n')
        f.write(f'extra_specs: {extra_specs}\n')
    
    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
