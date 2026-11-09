"""
Train DDQN-based active touch exploration on top of a frozen TactMAE model.

Usage:
    python exploration/train.py \
        --checkpoint /path/to/tactmae.ckpt \
        --config configs/shapenet_tactile_scratch.yaml \
        --exp_id my_run
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from octmae.nets import OctMAE
from octmae.utils.config import parse_config

from exploration.environment import TactMAEEnv
from exploration.data_loader import get_exploration_loader
from exploration import ddqn as ddqn_module
from exploration.replay import ReplayMemory


def load_tactmae(checkpoint_path, config):
    model = OctMAE(config)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    # strip Lightning prefix if present
    state = {k.replace('model.', '', 1): v for k, v in state.items() if k.startswith('model.')}
    model.load_state_dict(state, strict=False)
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Engine:
    def __init__(self, args):
        self.args = args
        self.steps = 0
        self.episode = 0
        self.epoch = 0
        self.epsilon = args.epsilon_start
        self.best_loss = float('inf')
        self.current_loss = float('inf')

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)

    def setup(self):
        config = parse_config(self.args.config)
        # inject exploration-specific args into config
        config.exploration_budget = self.args.budget
        config.max_touches = getattr(config, 'max_touches', 20)

        tactmae = load_tactmae(self.args.checkpoint, config)
        self.env = TactMAEEnv(tactmae, config, device='cuda')

        self.replay_memory = ReplayMemory(self.args)
        self.policy = ddqn_module.DDQN(self.args, self.replay_memory).cuda()
        self.target_net = ddqn_module.DDQN(self.args, replay=None).cuda()
        self.target_net.load_state_dict(self.policy.state_dict())
        self.target_net.eval()

        self.train_loader = get_exploration_loader(config, split='train', dataset_name=config.train_dataset_name)
        self.val_loader = get_exploration_loader(config, split='val', dataset_name=config.eval_dataset_name)
        self.test_loader = get_exploration_loader(config, split='test', dataset_name=config.eval_dataset_name)

        self.writer = SummaryWriter(os.path.join('experiments/tensorboard', self.args.exp_id))
        self.window = 1000
        self.ave_reward = torch.zeros(self.window)
        self.ave_recon = torch.zeros(self.window)

        self.resume()

    def run(self):
        self.setup()

        if self.args.do_eval:
            self.load(best=True)
            self.test()
            return

        for epoch in range(self.epoch, self.args.epochs):
            self.train()
            if self.steps >= self.args.burn_in:
                with torch.no_grad():
                    self.validate()
                self.check_and_save()
            self.epoch += 1

    # ------------------------------------------------------------------
    def _iter_batches(self, loader):
        """Accumulate individual scenes into env_batch_size batches."""
        buf = []
        for scene in loader:
            if isinstance(scene, list):
                buf.extend(scene)
            else:
                buf.append(scene)
            if len(buf) >= self.args.env_batch_size:
                yield buf[:self.args.env_batch_size]
                buf = buf[self.args.env_batch_size:]

    def train(self):
        step_count = 0
        for scenes in tqdm(self._iter_batches(self.train_loader), desc=f'Train E{self.epoch}'):
            if step_count >= self.args.train_steps:
                break

            obs = self.env.reset(scenes)
            all_done = False

            while not all_done:
                if self.steps >= self.args.burn_in:
                    self.epsilon = self.policy.update_epsilon(self.epsilon, self.args)

                give_random = self.steps < self.args.burn_in
                action = self.policy.get_action(obs, self.epsilon, give_random=give_random)

                with torch.no_grad():
                    next_obs, reward, all_done = self.env.step(action)

                self.policy.add_experience(action, obs, next_obs, reward)

                if self.steps >= self.args.burn_in:
                    self.policy.update_parameters(self.target_net)

                if self.steps % self.args.target_update == 0 and self.steps >= self.args.burn_in:
                    print('++ updating target ++')
                    self.target_net.load_state_dict(self.policy.state_dict())

                obs = next_obs
                self.steps += 1

            recon = float((obs['score'] / (obs['first_score'] + 1e-8)).mean().item())
            reward_val = float(((obs['first_score'] - obs['score']) / (obs['first_score'] + 1e-8)).mean().item())
            idx = self.episode % self.window
            self.ave_reward[idx] = reward_val
            self.ave_recon[idx] = recon
            n = min(self.episode + 1, self.window)
            tqdm.write(
                f'T E{self.epoch} ep{self.episode} recon={recon:.4f} '
                f'reward={reward_val:.4f} avg_r={self.ave_reward[:n].mean():.4f} '
                f'eps={self.epsilon:.3f}'
            )
            self.episode += 1
            step_count += 1

        if self.steps >= self.args.burn_in:
            n = min(self.episode, self.window)
            self.writer.add_scalar('train/recon', self.ave_recon[:n].mean(), self.steps)
            self.writer.add_scalar('train/reward', self.ave_reward[:n].mean(), self.steps)

    def test(self):
        print('*' * 30)
        print('Testing on test set')
        results = []
        for scenes in tqdm(self._iter_batches(self.test_loader), desc='Test'):
            obs = self.env.reset(scenes)
            all_done = False
            chosen_actions = []

            while not all_done:
                action = self.policy.get_action(obs, eps_threshold=-1, give_random=False)
                chosen_actions.append(action.tolist())
                with torch.no_grad():
                    obs, _, all_done = self.env.step(action)

            for i, scene in enumerate(scenes):
                results.append({
                    'key':         scene['key'],
                    'first_score': obs['first_score'][i].item(),
                    'final_score': obs['score'][i].item(),
                    'actions':     [a[i] for a in chosen_actions],
                })

        recon = sum(r['final_score'] / (r['first_score'] + 1e-8) for r in results) / len(results)
        reward = sum((r['first_score'] - r['final_score']) / (r['first_score'] + 1e-8) for r in results) / len(results)
        print(f'Test: recon={recon:.4f}  reward={reward:.4f}  n={len(results)}')

        save_path = os.path.join(self.args.results_dir, 'test_results.pt')
        torch.save(results, save_path)
        print(f'Results saved: {save_path}')
        print('*' * 30)

    def validate(self):
        print('*' * 30)
        scores_all = []
        step_count = 0
        for scenes in tqdm(self._iter_batches(self.val_loader), desc=f'Val E{self.epoch}'):
            if step_count >= self.args.valid_steps:
                break
            obs = self.env.reset(scenes)
            all_done = False
            cur_scores = [obs['score'].clone()]

            while not all_done:
                action = self.policy.get_action(obs, eps_threshold=-1, give_random=False)
                with torch.no_grad():
                    obs, _, all_done = self.env.step(action)
                cur_scores.append(obs['score'].clone())

            scores_all.append(torch.stack(cur_scores, dim=1))  # (B, T)
            step_count += 1

        scores_all = torch.cat(scores_all, dim=0)
        self.current_loss = (scores_all[:, -1] / (scores_all[:, 0] + 1e-8)).mean().item()
        rewards = ((scores_all[:, 0] - scores_all[:, -1]) / (scores_all[:, 0] + 1e-8)).mean().item()
        print(f'Val E{self.epoch}: recon={self.current_loss:.4f} reward={rewards:.4f} best={self.best_loss:.4f}')
        print('*' * 30)

        self.writer.add_scalar('val/recon', self.current_loss, self.steps)
        self.writer.add_scalar('val/reward', rewards, self.steps)
        self.writer.add_scalar('val/epsilon', self.epsilon, self.steps)

    def check_and_save(self):
        if self.current_loss < self.best_loss:
            improvement = self.best_loss - self.current_loss
            print(f'Saving best (CD improvement {improvement:.4f})')
            self.best_loss = self.current_loss
            self.save(best=True)
        self.save(best=False)
        self.replay_memory.save(os.path.join(self.args.checkpoint_dir, 'recent'))
        if self.epoch % self.args.snapshot_every == 0:
            self.save_snapshot(self.epoch)

    def save_snapshot(self, epoch):
        path = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch:04d}_model.pt')
        torch.save({
            'dqn_weights':    self.policy.state_dict(),
            'target_weights': self.target_net.state_dict(),
            'episode':        self.episode,
            'steps':          self.steps,
            'epsilon':        self.epsilon,
            'epoch':          self.epoch,
            'best_loss':      self.best_loss,
        }, path)
        self.replay_memory.save(os.path.join(self.args.checkpoint_dir, f'epoch_{epoch:04d}'))
        print(f'Snapshot saved: epoch_{epoch:04d}')

    def save(self, best=False):
        tag = 'best' if best else 'recent'
        path = os.path.join(self.args.checkpoint_dir, f'{tag}_model.pt')
        torch.save({
            'dqn_weights':    self.policy.state_dict(),
            'target_weights': self.target_net.state_dict(),
            'episode':        self.episode,
            'steps':          self.steps,
            'epsilon':        self.epsilon,
            'epoch':          self.epoch,
            'best_loss':      self.best_loss,
            'ave_reward':     self.ave_reward,
            'ave_recon':      self.ave_recon,
        }, path)

    def load(self, best=False):
        tag = 'best' if best else 'recent'
        path = os.path.join(self.args.checkpoint_dir, f'{tag}_model.pt')
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(ckpt['dqn_weights'])
        self.episode = ckpt['episode'] + 1
        if not self.args.do_eval:
            self.target_net.load_state_dict(ckpt['target_weights'])
            self.steps = ckpt['steps']
            self.epsilon = ckpt['epsilon']
            self.epoch = ckpt['epoch'] + 1
            self.best_loss = ckpt['best_loss']
            self.ave_reward = ckpt['ave_reward']
            self.ave_recon = ckpt['ave_recon']

    def resume(self):
        path = os.path.join(self.args.checkpoint_dir, 'recent_model.pt')
        if os.path.exists(path):
            print('Resuming from checkpoint')
            self.load(best=False)
            replay_path = os.path.join(self.args.checkpoint_dir, 'recent')
            if os.path.exists(replay_path + '_replay_buffer.pt'):
                self.replay_memory.load(replay_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to TactMAE checkpoint (.ckpt)')
    parser.add_argument('--config', type=str, required=True, help='Path to TactMAE config yaml')
    parser.add_argument('--exp_id', type=str, default='exploration_test')
    parser.add_argument('--do_eval', action='store_true', default=False)

    # DDQN hyperparams
    parser.add_argument('--env_batch_size', type=int, default=3, help='Number of scenes per RL episode')
    parser.add_argument('--budget', type=int, default=5, help='Touch budget per episode')
    parser.add_argument('--num_actions', type=int, default=20, help='Max touches in dataset')
    parser.add_argument('--latent_dim', type=int, default=192, help='TactMAE grid_features dim')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--normalization', type=str, default='first', choices=['first', 'current', 'none'])
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_decay', type=float, default=0.9999)
    parser.add_argument('--epsilon_end', type=float, default=0.01)
    parser.add_argument('--mem_capacity', type=int, default=500)
    parser.add_argument('--burn_in', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--target_update', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--snapshot_every', type=int, default=50, help='Save snapshot every N epochs')
    parser.add_argument('--train_steps', type=int, default=20)
    parser.add_argument('--valid_steps', type=int, default=20)

    # paths
    parser.add_argument('--checkpoint_dir', type=str, default='experiments/checkpoint/exploration')
    parser.add_argument('--results_dir', type=str, default='results/exploration')

    args = parser.parse_args()
    engine = Engine(args)
    engine.run()


if __name__ == '__main__':
    main()
