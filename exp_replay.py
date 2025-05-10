import numpy as np
import torch

class ExperienceReplay:
    def __init__(self, num_frame_stack=4, capacity=int(1e5), pic_size=(96, 96)):
        self.num_frame_stack = num_frame_stack
        self.capacity = capacity
        self.pic_size = pic_size
        self.counter = 0
        self.frame_window = None
        self.init_caches()
        self.expecting_new_episode = True

    def __len__(self):
        return min(self.counter, self.capacity)


    def add_experience(self, frame, action, done, reward):
        assert self.frame_window is not None, "Start episode first"
        self.counter += 1
        frame_idx = self.counter % self.max_frame_cache
        exp_idx = (self.counter - 1) % self.capacity

        self.prev_states[exp_idx] = self.frame_window.copy()
        self.frame_window = np.roll(self.frame_window, -1)
        self.frame_window[-1] = frame_idx
        self.next_states[exp_idx] = self.frame_window.copy()
        self.actions[exp_idx] = action
        self.is_done[exp_idx] = done
        self.frames[frame_idx] = frame
        self.rewards[exp_idx] = reward
        
        if done:
            self.expecting_new_episode = True

    def start_new_episode(self, frame):
        assert self.expecting_new_episode, "Previous episode didn't end yet"
        frame_idx = self.counter % self.max_frame_cache
        self.frame_window = np.repeat(frame_idx, self.num_frame_stack)
        self.frames[frame_idx] = frame
        self.expecting_new_episode = False

    def sample_mini_batch(self, n):
        count = min(self.capacity, self.counter)
        batchidx = np.random.randint(count, size=n)

        prev_frames = self.frames[self.prev_states[batchidx]]
        next_frames = self.frames[self.next_states[batchidx]]
        
        return {
            "reward": torch.FloatTensor(self.rewards[batchidx]),
            "prev_state": torch.FloatTensor(prev_frames),
            "next_state": torch.FloatTensor(next_frames),
            "actions": torch.LongTensor(self.actions[batchidx]),
            "done_mask": torch.BoolTensor(self.is_done[batchidx])
        }

    def current_state(self):
        assert self.frame_window is not None, "Do something first"
        sf = self.frames[self.frame_window]
        return torch.FloatTensor(sf).unsqueeze(0)

    def init_caches(self):
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.prev_states = -np.ones((self.capacity, self.num_frame_stack), dtype=np.int32)
        self.next_states = -np.ones((self.capacity, self.num_frame_stack), dtype=np.int32)
        self.is_done = np.zeros(self.capacity, dtype=np.bool_)
        self.actions = -np.ones(self.capacity, dtype=np.int32)
        self.max_frame_cache = self.capacity + 2 * self.num_frame_stack + 1
        self.frames = np.zeros((self.max_frame_cache, *self.pic_size), dtype=np.float32)