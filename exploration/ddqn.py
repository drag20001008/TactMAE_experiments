import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from exploration.model import QNetwork


class DDQN(nn.Module):
    def __init__(self, args, replay=None):
        super().__init__()
        self.args = args
        self.model = QNetwork(args).cuda()
        self.replay = replay
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def penalise_actions(self, values, obs):
        """Set Q-values of already-visited actions to -1e10."""
        values[obs['mask'].cuda() > 0] = -1e10
        return values

    def update_epsilon(self, epsilon, args):
        return max(args.epsilon_end, epsilon * args.epsilon_decay)

    def add_experience(self, action, observation, next_observation, reward):
        self.replay.push(action, observation, next_observation, reward)

    def forward(self, obs, next=False, penalize=True):
        value = self.model(obs, next=next)
        if penalize:
            mask_key = 'mask_n' if next else 'mask'
            value[obs[mask_key].cuda() > 0] = -1e10
        return value

    def get_action(self, obs, eps_threshold, give_random=False):
        if random.random() < eps_threshold or give_random:
            return self._random_action(obs['mask'])
        with torch.no_grad():
            self.model.eval()
            q_values = self(obs)
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def _random_action(self, mask):
        """Sample a uniformly random unvisited action per sample."""
        B = mask.shape[0]
        actions = []
        for i in range(B):
            available = (mask[i] == 0).nonzero(as_tuple=False).squeeze(1).tolist()
            actions.append(random.choice(available) if available else 0)
        return np.array(actions)

    def update_parameters(self, target_net):
        self.model.train()
        batch = self.replay.sample()
        if batch is None:
            return None

        not_done_mask = batch['mask'].cuda().sum(dim=1) < self.args.budget - 1
        actions = batch['actions'].cuda().long()
        rewards = batch['rewards'].cuda()
        first_score = batch['first_score'].cuda()

        if self.args.normalization == 'first':
            rewards = rewards / (first_score + 1e-8)
        elif self.args.normalization == 'current':
            cur_score = batch['score'].cuda()
            rewards = rewards / (cur_score + 1e-8)

        # DDQN update
        all_q_cur = self.forward(batch, penalize=False)
        q_values = all_q_cur.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            best_next = self.forward(batch, next=True).detach().max(1)[1]
            target_vals = target_net.forward(batch, next=True, penalize=False).detach()
            q_next = torch.zeros(q_values.shape[0], device='cuda')
            for i in range(q_values.shape[0]):
                if not_done_mask[i]:
                    q_next[i] = target_vals[i][best_next[i]]
            targets = self.args.gamma * q_next + rewards

        loss = ((q_values - targets) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def state_dict(self):
        return self.model.state_dict()
