import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import numpy as np
from config import CONFIG

class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_filters)
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
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
    def __init__(self, num_channels=256, num_res_blocks=7):
        super().__init__()
        self.conv_block = nn.Conv2d(in_channels=9, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_block_bn = nn.BatchNorm2d(num_features=num_channels)
        self.conv_block_act = nn.ReLU()

        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])

        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(num_features=16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 9 * 10, 2086)

        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 9 * 10, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)

        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = policy.reshape(-1, 16 * 10 * 9)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)

        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = value.reshape(-1, 8 * 10 * 9)
        value = self.value_fc1(value)
        value = self.value_act2(value)
        value = self.value_fc2(value)

        return policy, value

class PolicyValueNet:
    def __init__(self, model_file=None, use_gpu=True, device='cuda'):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))

    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    def policy_value_fn(self, board):
        self.policy_value_net.eval()
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(-1, 9, 10, 9)).astype('float16')
        current_state = torch.as_tensor(current_state).to(self.device)

        with autocast():
            log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy().astype('float16').flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value.detach().numpy()

    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()

        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)

        self.optimizer.zero_grad()

        for params in self.optimizer.param_groups:
            params['lr'] = lr

        log_act_probs, value = self.policy_value_net(state_batch)
        value = value.reshape(-1)

        value_loss = F.mse_loss(input=value, target=winner_batch)

        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))

        loss = value_loss + policy_loss

        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()

if __name__ == '__main__':
    net = Net().to('cuda')
    test_data = torch.ones([8, 9, 10, 9]).to('cuda')
    x_act, x_val = net(test_data)
    print(x_act.shape) # torch.Size([8, 2086])
    print(x_val.shape) # torch.Size([8, 1])
