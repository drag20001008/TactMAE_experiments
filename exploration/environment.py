import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from ocnn.octree import Points

from octmae.nets.utils import get_xyz_from_octree
from octmae.utils.misc import normalize_pts


def chamfer_distance(pred, gt, max_pts=5000):
    if pred.shape[0] == 0:
        return torch.tensor(1e6, device=gt.device)
    if pred.shape[0] > max_pts:
        pred = pred[torch.randperm(pred.shape[0], device=pred.device)[:max_pts]]
    if gt.shape[0] > max_pts:
        gt = gt[torch.randperm(gt.shape[0], device=gt.device)[:max_pts]]
    dist = torch.cdist(pred.unsqueeze(0), gt.unsqueeze(0))[0]  # (N, M)
    return dist.min(1)[0].mean() + dist.min(0)[0].mean()


class TactMAEEnv:
    """
    RL environment for active touch exploration using a frozen TactMAE model.
    Actions select from up to max_touches pre-computed touches per scene.
    No real-time simulation — touches are pre-computed in the dataset.
    """

    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.num_actions = getattr(config, 'max_touches', 20)
        self.budget = getattr(config, 'exploration_budget', 5)
        self.grid_size = config.grid_size
        self.min_lod = config.min_lod
        self.max_lod = config.max_lod
        self.grid_res = 1 << self.min_lod

        self.model.eval()
        self.model.update_octree = True  # use predicted octree for inference

        self._grid_features = None
        self._voxel_counts = None
        self.model.mae_encoder.register_forward_hook(self._hook_fn)

        self.scenes = None
        self.touch_mask = None
        self.first_latent = None
        self.first_score = None
        self._current_score = None
        # per-scene available-action mask (1 = touch exists for this scene)
        self._avail_mask = None

    def _hook_fn(self, module, inputs, output):
        self._grid_features = output.detach()
        self._voxel_counts = inputs[2]  # list of per-sample voxel counts

    def _extract_latent(self):
        features = self._grid_features
        counts = self._voxel_counts
        dim = features.shape[1]
        latents = []
        start = 0
        for c in counts:
            if c > 0:
                latents.append(features[start:start + c].max(dim=0)[0])
            else:
                latents.append(torch.zeros(dim, device=features.device))
            start += c
        return torch.stack(latents, dim=0)  # (B, dim_mae)

    def _build_batch(self, touch_mask):
        rgbs, masks, depths, Ks, z_mins = [], [], [], [], []
        depth_pts_list, tactile_pts_list, gt_pts_list = [], [], []
        keys = []

        for i, scene in enumerate(self.scenes):
            rgbs.append(scene['rgb'])
            masks.append(scene['mask'])
            depths.append(scene['depth'])
            Ks.append(scene['K'])
            z_min_val = scene['z_min']
            if not torch.is_tensor(z_min_val):
                z_min_val = torch.tensor(z_min_val, dtype=torch.float32)
            z_mins.append(z_min_val)
            depth_pts_list.append(scene['depth_pts'].to(self.device))
            gt_pts_list.append(scene['gt_pts'].to(self.device))
            keys.append(scene['key'])

            tac_pts = None
            if scene['all_touch_pts'] is not None:
                selected = touch_mask[i].nonzero(as_tuple=False).squeeze(1).tolist()
                valid = [idx for idx in selected if idx < scene['n_touches']]
                if valid:
                    raw = scene['all_touch_pts'][valid].reshape(-1, 3).to(self.device)
                    z_min_d = z_min_val.to(self.device)
                    norm = normalize_pts(raw.clone(), z_min_d, self.grid_size, self.grid_res)
                    tac_pts = Points(norm)
                    tac_pts.clip()
            tactile_pts_list.append(tac_pts)

        rgb = torch.stack(rgbs).to(self.device)
        mask = torch.stack(masks).to(self.device)
        depth = torch.stack(depths).to(self.device)
        K = torch.stack(Ks).to(self.device)
        z_min = torch.stack(z_mins).to(self.device)

        return (rgb, mask, depth, depth_pts_list, tactile_pts_list, gt_pts_list, K, z_min, keys)

    def _run_model_and_score(self, touch_mask):
        batch = self._build_batch(touch_mask)
        gt_pts_list = batch[5]
        with torch.no_grad():
            output = self.model(batch)
        latent = self._extract_latent()
        octrees_out = output['octrees_out']

        scores = []
        pred_xyz = get_xyz_from_octree(octrees_out, self.max_lod)
        counts = octrees_out.batch_nnum[self.max_lod].tolist()
        start = 0
        for j, c in enumerate(counts):
            pred = pred_xyz[start:start + c]
            gt = gt_pts_list[j].points
            scores.append(chamfer_distance(pred, gt))
            start += c
        score = torch.stack(scores)
        return latent, score

    def reset(self, scene_list):
        self.scenes = scene_list
        B = len(scene_list)
        self.touch_mask = torch.zeros(B, self.num_actions, dtype=torch.float32)

        # mark non-existent touches as already-used (will be penalized by DDQN)
        self._avail_mask = torch.zeros(B, self.num_actions, dtype=torch.float32)
        for i, scene in enumerate(scene_list):
            n = scene['n_touches']
            self._avail_mask[i, :n] = 1.0
            if n < self.num_actions:
                self.touch_mask[i, n:] = 1.0  # mark unavailable as visited

        latent, score = self._run_model_and_score(self.touch_mask)
        self.first_latent = latent
        self.first_score = score
        self._current_score = score.clone()

        return {
            'mask':         self.touch_mask.clone(),
            'latent':       latent,
            'first_latent': self.first_latent,
            'score':        score,
            'first_score':  self.first_score,
        }

    def step(self, actions):
        """
        actions: array/list of B ints, each selecting a touch index.
        Returns (next_obs, reward, all_done).
        """
        B = len(self.scenes)
        prev_score = self._current_score.clone()

        for i in range(B):
            self.touch_mask[i, int(actions[i])] = 1.0

        latent, score = self._run_model_and_score(self.touch_mask)
        reward = prev_score - score  # positive = CD improved
        self._current_score = score.clone()

        available = self._avail_mask.sum(dim=1)  # actual touches per scene
        used_valid = (self.touch_mask * self._avail_mask).sum(dim=1)
        all_done = bool((used_valid >= self.budget).all() or
                        (used_valid >= available).all())

        obs = {
            'mask':         self.touch_mask.clone(),
            'latent':       latent,
            'first_latent': self.first_latent,
            'score':        score,
            'first_score':  self.first_score,
        }
        return obs, reward, all_done
