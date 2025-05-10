import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from exp_replay import ExperienceReplay  # THIS WAS MISSING

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 8, kernel_size=7, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 400),
            nn.ReLU(),
            nn.Linear(400, num_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, env, action_map, pic_size=(96, 96), num_frame_stack=3,
                 gamma=0.95, lr=1e-3, batch_size=64, buffer_size=int(1e5),
                 target_update_freq=1000, eps_start=1.0, eps_end=0.05,
                 eps_decay=100000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.num_frame_stack = num_frame_stack
        
        input_shape = (num_frame_stack, *pic_size)
        self.policy_net = DQN(input_shape, len(action_map)).to(self.device)
        self.target_net = DQN(input_shape, len(action_map)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ExperienceReplay(num_frame_stack, buffer_size, pic_size)
        self.action_map = action_map
        
        self.steps_done = 0
        self.episode_count = 0

    def select_action(self, state):
        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state.to(self.device)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.choice(len(self.action_map))]], 
                              device=self.device, dtype=torch.long)

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample_mini_batch(self.batch_size)
        state_batch = batch['prev_state'].to(self.device)
        next_state_batch = batch['next_state'].to(self.device)
        # Unsqueeze action_batch so that it becomes (batch_size, 1)
        action_batch = batch['actions'].to(self.device).unsqueeze(1)
        reward_batch = batch['reward'].to(self.device)
        done_batch = batch['done_mask'].to(self.device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma * ~done_batch) + reward_batch
        
        loss = nn.MSELoss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
