import os
from utils import *
import numpy as np
import torch
import torch.nn as nn
from typing import *
import sys
import argparse
import json
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        # 创建一个足够长的 PE 矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # 添加位置编码到输入上
        x = x + self.pe[:x.size(1), :]
        return x

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
        # Attention/Transformer 参数
        embed_dim = self.extra_specs.get('embed_dim', 64)
        num_heads = self.extra_specs.get('num_heads', 4)
        num_layers = self.extra_specs.get('num_layers', 2)

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
        elif model_type == "cnn":
            depth = self.extra_specs.get('depth', 4)
            channels = self.extra_specs.get('channels', 64)
            use_batch_norm = self.extra_specs.get('batch_norm', False)

            layers = []
            in_channels = 1
            def conv_block(in_ch, out_ch):
                block = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
                if use_batch_norm:
                    block.append(nn.BatchNorm2d(out_ch))
                block.append(nn.ReLU())
                return block
            
            for i in range(depth):
                out_channels = channels if i == 0 else channels
                layers.extend(conv_block(in_channels, out_channels))
                in_channels = out_channels
            
            self.conv_blocks = nn.Sequential(*layers)
            flat_features = channels * board_size * board_size
            self.fc = nn.Linear(flat_features, board_size * board_size)
        elif model_type == "attention":
            # 混合架构：CNN 提取局部特征 -> Self-Attention 整合全局特征
            # 浅层 CNN 提取 Token 特征
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, embed_dim, kernel_size=3, padding=1), nn.ReLU()
            )
            # 位置编码
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=board_size*board_size)
            # Multi-Head Attention 层
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            # 输出头
            self.output_head = nn.Linear(embed_dim, 1)
        elif model_type == "transformer":
            # 纯 Transformer 架构
            # Input Embedding
            self.embedding = nn.Sequential(
                nn.Conv2d(1, embed_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
            # 位置编码
            self.pos_encoder = PositionalEncoding(embed_dim, max_len=board_size*board_size)
            # Transformer Encoder Stack
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            # 输出头
            self.output_head = nn.Linear(embed_dim, 1)
        else:
            raise NotImplementedError("Model type not implemented.")

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
        input_board = output.clone()
        batch_size = output.size(0)

        logits = None

        if self.model_type in ["default", "cnn"]:
            # CNN 逻辑
            out = self.conv_blocks(output)
            out = out.view(out.size(0), -1)
            logits = self.fc(out)

        elif self.model_type == "attention":
            # CNN 提取特征
            features = self.feature_extractor(output)
            # 展平为序列
            # N*N = 144
            features = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)
            features = self.pos_encoder(features)
            # Self-Attention
            attn_output, _ = self.attention(features, features, features)
            # Residual + Norm
            features = self.norm(features + attn_output)
            # 输出头
            logits = self.output_head(features)
            # Squeeze
            logits = logits.squeeze(-1)

        elif self.model_type == "transformer":
            # Embedding: (B, 1, N, N) -> (B, Embed, N, N)
            features = self.embedding(output)
            # Flatten
            features = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)
            # Positional Encoding
            features = self.pos_encoder(features)
            # Transformer Encoder
            features = self.transformer_encoder(features)
            # Output Head -> (B, N*N)
            logits = self.output_head(features).squeeze(-1)

        # Masking & Softmax (所有模型通用)
        flat_board = input_board.view(batch_size, -1)
        illegal_mask = (flat_board != 0)
        logits = logits.masked_fill(illegal_mask, -1e9)
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

    def __init__(self, board_size: int, lr=1e-4, model_type: str = "default", action_injection: str = "none", extra_specs: dict = None):
        super().__init__()
        self.board_size = board_size
        self.model_type = model_type
        self.action_injection = action_injection
        self.extra_specs = extra_specs or {}
        
        # Define your NN structures here as the same. Torch modules have to be registered during the initialization
        # process.

        # BEGIN YOUR CODE
        # Determine input channels based on action injection
        if action_injection == "early":
            in_channels = 2  # board + action mask
        else:
            in_channels = 1
        
        if model_type == "default":
            use_deep = self.extra_specs.get('use_deep', False)
            
            if not use_deep:
                # Baseline CNN (3-Layer)
                self.conv_blocks = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU()
                )
                flat_features = 128 * board_size * board_size
            else:
                # Deep CNN (5-Layer)
                self.conv_blocks = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
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
            
            # For "fc" injection, add action coordinates to features
            if action_injection == "fc":
                self.fc = nn.Linear(flat_features + 2, board_size * board_size)
            else:
                self.fc = nn.Linear(flat_features, board_size * board_size)
                
        elif model_type in ["cnn", "attention", "transformer"]:
            # Get architecture specs
            depth = self.extra_specs.get('depth', 3)
            channels = self.extra_specs.get('channels', 64)  # Changed default from 32 to 64
            use_batch_norm = self.extra_specs.get('batch_norm', False)

            # Build convolutional layers
            layers = []
            current_channels = in_channels

            for i in range(depth):
                # Use fixed channels as specified, not exponentially growing
                out_ch = channels
                layers.append(nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1))
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU())

                # For "late" injection, add action before the last conv layer
                if action_injection == "late" and i == depth - 2 and i >= 0:  # Add before last conv
                    current_channels = out_ch + 1  # Add 1 channel for action
                else:
                    current_channels = out_ch

            self.conv_blocks = nn.Sequential(*layers)

            # Calculate features for FC layer
            final_channels = channels
            flat_features = final_channels * board_size * board_size

            # For "fc" injection, concatenate action index to features
            if action_injection == "fc":
                self.fc = nn.Linear(flat_features + 2, board_size * board_size)  # +2 for (x, y) action
            else:
                self.fc = nn.Linear(flat_features, board_size * board_size)
        else:
            raise NotImplementedError(f"Model type '{model_type}' not implemented for Critic.")
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
        # Convert action to tensor
        if isinstance(action, np.ndarray):
            action_tensor = torch.from_numpy(action).to(device).float()
        else:
            action_tensor = action.to(device).float()
        
        batch_size = output.size(0)
        
        # Early injection: add action as extra input channel
        if self.action_injection == "early":
            # Create action mask: (B, 1, N, N) with 1 at action position
            action_mask = torch.zeros(batch_size, 1, self.board_size, self.board_size).to(device)
            for i, (ax, ay) in enumerate(action_tensor):
                action_mask[i, 0, int(ax), int(ay)] = 1.0
            # Concatenate board and action mask
            output = torch.cat([output, action_mask], dim=1)  # (B, 2, N, N)
            out = self.conv_blocks(output)
        
        # Late injection: inject action before last conv layer
        elif self.action_injection == "late":
            # Process through all but last conv layer
            # Count the number of layers: each conv block has [Conv2d, (BatchNorm2d), ReLU]
            # We want to inject before the final conv layer
            layers_list = list(self.conv_blocks)
            # Find the last Conv2d layer
            last_conv_idx = -1
            for i in range(len(layers_list) - 1, -1, -1):
                if isinstance(layers_list[i], nn.Conv2d):
                    last_conv_idx = i
                    break
            
            # Process layers before the last conv
            for i in range(last_conv_idx):
                output = layers_list[i](output)
            
            # Create action mask and inject
            action_mask = torch.zeros(batch_size, 1, self.board_size, self.board_size).to(device)
            for j, (ax, ay) in enumerate(action_tensor):
                action_mask[j, 0, int(ax), int(ay)] = 1.0
            # Concatenate features with action mask before last conv
            output = torch.cat([output, action_mask], dim=1)
            
            # Process remaining layers (last conv and any following layers)
            for i in range(last_conv_idx, len(layers_list)):
                output = layers_list[i](output)
            out = output
        
        # FC injection or no injection
        else:
            out = self.conv_blocks(output)
        
        # Flatten features
        out = out.view(out.size(0), -1)
        
        # FC injection: concatenate action coordinates to features
        if self.action_injection == "fc":
            # Normalize action coordinates to [0, 1]
            action_coords = action_tensor / self.board_size
            out = torch.cat([out, action_coords], dim=1)
        
        # Generate Q-values for all positions
        q_values_all = self.fc(out)
        
        # Extract Q-values for specified actions
        output = q_values_all.gather(1, indices.unsqueeze(1))
        output = output.squeeze(1)
        # END YOUR CODE

        return output


class GobangModel(nn.Module):
    """
    The GobangModel class integrates the Actor and Critic classes for computation and training. Given state tensors "x"
    and action tensors "action", it directly outputs self.actor(x) and self.critic(x, action) as the policy and Q-values
    respectively.
    """
    def __init__(self, board_size: int, bound: int, model_type: str = "default", extra_specs: dict = None, lr: float = 1e-4):
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
        extra_specs = extra_specs or {}
        action_injection = extra_specs.get('action_injection', 'none')
        self.actor = Actor(board_size=board_size, lr=lr, model_type=model_type, extra_specs=extra_specs)
        self.critic = Critic(board_size=board_size, lr=lr, model_type=model_type, action_injection=action_injection, extra_specs=extra_specs)
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
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimizers (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.5, help='discount factor for training (default: 0.5)')
    parser.add_argument('--reward-type', type=str, default='default', dest='reward_type', 
                        help='reward type: default, heuristic (default: default)')
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
                    "lr": args.lr,
                    "gamma": args.gamma,
                    "reward_type": args.reward_type,
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
        
    agent = GobangModel(board_size=12, bound=5, model_type=args.model_type, extra_specs=extra_specs, lr=args.lr).to(device)
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
    train_model(agent, num_episodes=num_episodes, checkpoint=checkpoint, gamma=args.gamma, save_dir=save_folder, reward_type=args.reward_type)

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
        f.write(f'lr: {args.lr}\n')
        f.write(f'gamma: {args.gamma}\n')
        f.write(f'reward_type: {args.reward_type}\n')

    # If wandb_name is provided, create a symbolic link with that name
    if args.wandb_name:
        import re
        # Clean the wandb_name to be a valid directory name
        clean_name = re.sub(r'[^\w\-_]', '_', args.wandb_name)
        link_path = f'checkpoints/{clean_name}'

        # Remove the link if it already exists
        if os.path.islink(link_path):
            os.unlink(link_path)
            print(f"Removed existing symbolic link: {link_path}")
        elif os.path.exists(link_path):
            # If it's a directory, don't overwrite it
            print(f"Warning: {link_path} already exists and is not a symbolic link. Skipping link creation.")
        
        # Create the symbolic link if path is now available
        if not os.path.exists(link_path):
            # Create the symbolic link - use relative path to avoid issues
            # Get the relative path from the checkpoints directory to the save_folder
            rel_path = os.path.relpath(save_folder, os.path.dirname(link_path))
            os.symlink(rel_path, link_path)
            print(f"Created symbolic link from {link_path} to {rel_path}")

    if args.use_wandb:
        try:
            import wandb
            wandb.finish()
        except:
            pass
