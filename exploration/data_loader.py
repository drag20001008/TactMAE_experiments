import io
import random

import numpy as np
import torch
import webdataset as wds
import imageio.v3 as iio
from torchvision import transforms
from ocnn.octree import Points

from octmae.utils.misc import (
    rle_to_binary_mask, get_camera_rays, normalize_pts, decode_depth
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTRINSICS_K = {
    'shapenet': [[572.0, 0., 320.0], [0., 572.0, 240.0], [0., 0., 1.]],
}


def make_exploration_sample_wrapper(config,
                                    K=[[572.0, 0., 320.0], [0., 572.0, 240.0], [0., 0., 1.]]):
    """
    Active-3D의 mesh_loader_active와 동일한 역할.
    make_sample_wrapper와 달리, touch를 랜덤 샘플링하지 않고
    available한 모든 touch를 그대로 반환한다.
    """
    img_size = (config.img_height, config.img_width)
    resized_img_size = (480, 640)
    grid_size = config.grid_size
    min_lod = config.min_lod
    grid_res = 1 << min_lod
    K_arr = np.asarray(K, dtype=np.float32)
    camera_rays = get_camera_rays(K_arr, img_size)

    transform = transforms.Compose([
        transforms.Resize(resized_img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    def make_sample(sample):
        rgb = transform(sample['rgb.jpg'])
        depth = sample['camera.json']['depth_scale'] * sample['depth.png']
        depth = depth.astype(np.float32)

        K_sample = np.asarray(sample['camera.json']['cam_K']).astype(np.float32).reshape(3, 3)
        obj_pose = sample['gt.json']
        obj_info = sample['gt_info.json']
        mask_rle = sample['mask_visib.json']

        mask = np.zeros(img_size, dtype=bool)
        for ind, (oi, op) in enumerate(zip(obj_info, obj_pose)):
            if oi['visib_fract'] < 0.2:
                continue
            imask = rle_to_binary_mask(mask_rle[str(ind)], oi['bbox_visib'])
            mask = np.logical_or(mask, imask)
        mask = np.logical_and(mask, depth > 10.0)

        depth_pts_raw = torch.from_numpy(
            (camera_rays * depth[:, :, None])[mask > 0.0].astype(np.float32)
        ).reshape(-1, 3)

        spc = torch.from_numpy(sample['spc.npz']['spc'].astype(np.float32))
        obj_ids = torch.from_numpy(sample['spc.npz']['obj_ids'].astype(np.int32))
        _, obj_ids = torch.unique(obj_ids, sorted=True, return_inverse=True)

        vdb_grid_size = 1.25
        offset = vdb_grid_size * 0.5
        spc_mask = torch.logical_and(spc[:, 3] < vdb_grid_size, spc[:, 3] > -vdb_grid_size)
        spc = spc[spc_mask]
        obj_ids = obj_ids[spc_mask]

        # z_min: depth_pts 기준으로 계산
        z_min = (depth_pts_raw[:, 2].min() // grid_size) * grid_size - 5 * grid_size

        pts_3d_depth = Points(normalize_pts(depth_pts_raw.clone(), z_min, grid_size, grid_res))
        pts_3d_depth.clip()

        pts_3d_gt = Points(
            points=normalize_pts(spc[:, :3] + offset, z_min, grid_size, grid_res),
            normals=spc[:, 4:7],
            features=spc[:, 3:4],
            labels=obj_ids
        )
        pts_3d_gt.clip()

        # 모든 touch를 raw tensor로 저장 (normalize는 환경에서)
        all_touch_pts = None
        n_touches = 0
        if 'tactile.npz' in sample and config.use_tactile:
            tactile_raw = torch.from_numpy(sample['tactile.npz']['pts'].astype(np.float32))
            pts_per_touch = 200
            n_touches = tactile_raw.shape[0] // pts_per_touch
            if n_touches > 0:
                all_touch_pts = tactile_raw.reshape(n_touches, pts_per_touch, 3)

        if depth_pts_raw.shape[0] < 100 or pts_3d_gt.points.shape[0] < 1000:
            raise Exception('Not enough points', sample['__key__'])

        return {
            'rgb':           rgb,
            'depth':         torch.from_numpy(depth),
            'mask':          torch.from_numpy(np.ascontiguousarray(mask)),
            'K':             torch.from_numpy(K_sample),
            'z_min':         z_min,
            'depth_pts':     pts_3d_depth,   # 정규화된 Points (변하지 않음)
            'gt_pts':        pts_3d_gt,       # 정규화된 Points (GT)
            'all_touch_pts': all_touch_pts,   # (n_touches, 200, 3) raw, None이면 touch 없음
            'n_touches':     n_touches,
            'key':           sample['__key__'],
        }

    return make_sample


def collate_exploration(batch):
    return batch  # 각 scene을 list로 유지 (touch pts가 sample마다 다르기 때문)


def get_exploration_loader(config, split='train', dataset_name='shapenet'):
    if split == 'train':
        url = config.train_dataset_url
    elif split == 'val':
        url = config.val_dataset_url
    else:
        url = config.test_dataset_url

    K = INTRINSICS_K[dataset_name]
    dataset = (
        wds.WebDataset(url, handler=wds.warn_and_continue, shardshuffle=(split == 'train'))
        .decode(decode_depth, 'pil', handler=wds.warn_and_continue)
        .map(make_exploration_sample_wrapper(config, K=K), handler=wds.warn_and_continue)
    )
    loader = wds.WebLoader(dataset, batch_size=None, shuffle=False, num_workers=2,
                           collate_fn=collate_exploration)
    return loader
