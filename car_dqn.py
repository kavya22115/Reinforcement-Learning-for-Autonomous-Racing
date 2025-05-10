import torch
import numpy as np

class CarRacingDQN:
    def __init__(self, env, max_negative_rewards=100):
        self.all_actions = torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 0.5],
            [0, 0, 0],
            [1, 0, 0]
        ], dtype=torch.float32)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim_actions = len(self.all_actions)
        
        self.gas_actions = (self.all_actions[:, 1] == 1) & (self.all_actions[:, 2] == 0)
        self.break_actions = self.all_actions[:, 2] > 0
        self.n_gas_actions = self.gas_actions.sum().item()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards
        self.env = env

    def get_random_action(self):
        action_weights = 14.0 * self.gas_actions.numpy() + 1.0
        action_weights /= action_weights.sum()
        return np.random.choice(self.dim_actions, p=action_weights)

    def check_early_stop(self, reward, totalreward, fie):
        if reward < 0 and fie > 10:
            self.neg_reward_counter += 1
            done = (self.neg_reward_counter > self.max_neg_rewards)
            punishment = -20.0 if done and totalreward <= 500 else 0.0
            if done:
                self.neg_reward_counter = 0
            return done, punishment
        self.neg_reward_counter = 0
        return False, 0.0