import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from config import CONFIG
from game import Board

class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                               kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)

class Net(nn.Module):
    """
    shape: (N, C=9, H=10, W=9)
    """
    def __init__(self, num_channels=256, num_res_blocks=7):
        super().__init__()
        self.conv_block = nn.Conv2d(in_channels=9, out_channels=num_channels,
                                    kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_block_bn = nn.BatchNorm2d(num_features=num_channels)
        self.conv_block_act = nn.ReLU()

        self.res_blocks = nn.ModuleList(
            [ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])

        # 策略网络: 用于输出所有合法走子的概率
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16,
                                     kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(num_features=16)
        self.policy_act = nn.ReLU()
        # 2086 对应 get_all_legal_moves 能够获得的所有的 2086 个合法走子的概率
        self.policy_fc = nn.Linear(16 * 9 * 10, 2086)

        # 价值网络: 用于输出当前盘面得分
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8,
                                    kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 9 * 10, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)

        # 策略网络的输出
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = policy.reshape(-1, 16 * 10 * 9)
        policy = self.policy_fc(policy)
        # 通过 softmax 计算概率, 使用 log_softmax 为了提高数值稳定性
        log_probs = F.log_softmax(policy)

        # 价值网络的输出
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = value.reshape(-1, 8 * 10 * 9)
        value = self.value_fc1(value)
        value = self.value_act2(value)
        value = self.value_fc2(value)
        # 盘面得分在 (-1, 1) 之间
        score = F.tanh(value)

        return log_probs, score

class PolicyValueNet:
    def __init__(self, model_file=None, use_gpu=True, device='cuda'):
        # 是否使用 GPU
        self.use_gpu = use_gpu
        # 使用设备
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-3)
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))

    def policy_value(self, state_batch):
        """
        返回 批盘面 的所有合法走子概率和得分
        """
        # eval 模式, 对 BatchNorm 等模块有影响
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).to(self.device)
        log_probs, score = self.policy_value_net(state_batch)
        log_probs, score = log_probs.cpu(), score.cpu()
        probs = np.exp(log_probs.detach().numpy())
        return probs, score.detach().numpy()

    def policy_value_fn(self, board: Board):
        # eval 模式, 对 BatchNorm 等模块有影响
        self.policy_value_net.eval()
        # 获取当前盘面的合法走子 list
        moves_id_list = board.availables
        # board.current_state 为当前盘面, np.ascontiguousarray 为保证当前张量按 C 语言顺序在内存中存储
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 9, 10, 9)).astype('float16')
        current_state = torch.as_tensor(current_state).to(self.device)

        with autocast():
            log_probs, score = self.policy_value_net(current_state)
        log_probs, score = log_probs.cpu(), score.cpu()
        # flatten 将数据拉成一维
        probs = np.exp(log_probs.detach().numpy().astype('float16').flatten())
        # moves_id2probs 为走子 id 到概率的映射
        moves_id2probs = zip(moves_id_list, probs[moves_id_list])
        return moves_id2probs, score.detach().numpy()

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        # train 模式, 对 BatchNorm 等模块有影响
        self.policy_value_net.train()

        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)

        # 梯度清零
        self.optimizer.zero_grad()

        for params in self.optimizer.param_groups:
            # 更新学习率
            params['lr'] = lr

        log_probs, score = self.policy_value_net(state_batch)
        # score 被拉成一维
        score = score.reshape(-1)

        value_loss = F.mse_loss(input=score, target=winner_batch)

        # 对 1 维求和计算 NLLLoss
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_probs, dim=1))

        loss = value_loss + policy_loss

        # 反向传播, 获取梯度
        loss.backward()
        # 通过梯度优化参数
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=1))
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

if __name__ == '__main__':
    net = Net().to('cuda')
    test_data = torch.ones([8, 9, 10, 9]).to('cuda')
    x_act, x_val = net(test_data)
    print(x_act.shape) # torch.Size([8, 2086])
    print(x_val.shape) # torch.Size([8, 1])

    print(torch.sum(x_act * x_act, dim=1).shape)
