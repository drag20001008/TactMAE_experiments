import os

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, args):
        self.args = args
        cap = args.mem_capacity
        na = args.num_actions
        ld = args.latent_dim  # 192 for TactMAE

        self.mask = torch.zeros(cap, na)
        self.mask_n = torch.zeros(cap, na)
        self.actions = torch.zeros(cap)
        self.rewards = torch.zeros(cap)
        self.score = torch.zeros(cap)
        self.score_n = torch.zeros(cap)
        self.first_score = torch.zeros(cap)
        self.latent = torch.zeros(cap, ld)
        self.latent_n = torch.zeros(cap, ld)
        self.first_latent = torch.zeros(cap, ld)

        self.position = 0
        self.count_seen = 0

    def push(self, action, observation, next_observation, reward):
        B = len(action)
        for i in range(B):
            self.actions[self.position] = int(action[i])
            self.rewards[self.position] = reward[i].item() if torch.is_tensor(reward[i]) else float(reward[i])
            self.score[self.position] = observation['score'][i].item()
            self.score_n[self.position] = next_observation['score'][i].item()
            self.first_score[self.position] = observation['first_score'][i].item()
            self.mask[self.position] = observation['mask'][i].cpu()
            self.mask_n[self.position] = next_observation['mask'][i].cpu()
            self.latent[self.position] = observation['latent'][i].cpu()
            self.latent_n[self.position] = next_observation['latent'][i].cpu()
            self.first_latent[self.position] = observation['first_latent'][i].cpu()
            self.count_seen += 1
            self.position = (self.position + 1) % self.args.mem_capacity

    def sample(self):
        if self.count_seen < self.args.burn_in or self.count_seen < self.args.train_batch_size:
            return None
        indices = np.random.choice(
            min(self.count_seen, self.args.mem_capacity), self.args.train_batch_size, replace=False
        )
        return {
            'mask':         self.mask[indices],
            'mask_n':       self.mask_n[indices],
            'actions':      self.actions[indices],
            'rewards':      self.rewards[indices],
            'score':        self.score[indices],
            'score_n':      self.score_n[indices],
            'first_score':  self.first_score[indices],
            'latent':       self.latent[indices],
            'latent_n':     self.latent_n[indices],
            'first_latent': self.first_latent[indices],
        }

    def save(self, directory):
        data = {k: getattr(self, k) for k in
                ['mask', 'mask_n', 'actions', 'rewards', 'score', 'score_n',
                 'first_score', 'latent', 'latent_n', 'first_latent',
                 'position', 'count_seen']}
        temp = directory + '_replay_buffer_temp.pt'
        full = directory + '_replay_buffer.pt'
        torch.save(data, temp)
        os.rename(temp, full)

    def load(self, directory):
        data = torch.load(directory + '_replay_buffer.pt')
        for k, v in data.items():
            setattr(self, k, v)
