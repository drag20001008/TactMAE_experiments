"""
Scan training shards to find samples where VoxelOcclusionTester returns all-False
(no occupied voxels), which causes F.cross_entropy(empty, empty) = NaN.

Lightweight: skips SPC/VDB, loads only depth+mask+camera per sample.

Usage:
  python scripts/scan_vot_empty.py \
      --config configs/shapenet_no_tactile_scratch.yaml \
      --out_dir vot_empty_vis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import io
import glob
import argparse
import tarfile

import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from vot import VoxelOcclusionTester
from octmae.utils.config import parse_config
from octmae.utils.misc import rle_to_binary_mask

SHARD_PATTERN = '/nas1/1000_Members/drag1008/TactMAE/train/shapenet_*.tar'


def build_vot(config):
    vot_sf  = getattr(config, 'vot_scale_factor', 2)
    vot_inv = 1.0 / vot_sf
    vot = VoxelOcclusionTester(
        round(config.img_height * vot_inv),
        round(config.img_width  * vot_inv),
        config.grid_size,
    ).cuda()
    vot.eval()
    return vot, vot_sf, vot_inv


def process_raw_sample(raw, config, vot, vot_sf, vot_inv):
    """
    raw: dict with keys depth.png, camera.json, mask_visib.json, gt_info.json
    Returns (occ_count, n_depth, z_min_val, mask_px, depth_min, depth_max)
    or raises on bad data.
    """
    cam         = raw['camera.json']
    depth_scale = cam['depth_scale']
    K_np        = np.asarray(cam['cam_K'], dtype=np.float32).reshape(3, 3)
    depth_np    = raw['depth.png'] * depth_scale    # (H, W) float32

    H, W = depth_np.shape

    # Build mask (same logic as make_sample_wrapper)
    gt_info  = raw['gt_info.json']
    mask_rle = raw['mask_visib.json']
    mask_np  = np.zeros((H, W), dtype=bool)
    for ind, oi in enumerate(gt_info):
        if oi.get('visib_fract', 0.0) < 0.2:
            continue
        rle = mask_rle.get(str(ind))
        if rle is not None:
            imask = rle_to_binary_mask(rle, oi.get('bbox_visib'))
            mask_np = np.logical_or(mask_np, imask)
    mask_np = np.logical_and(mask_np, depth_np > 10.0)

    # Depth pts for z_min
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays  = np.stack([(u - K_np[0, 2]) / K_np[0, 0],
                      (v - K_np[1, 2]) / K_np[1, 1],
                      np.ones((H, W), dtype=np.float32)], axis=-1)
    depth_pts = (rays * depth_np[:, :, None])[mask_np].astype(np.float32)

    if depth_pts.shape[0] < 10:
        raise ValueError('not enough depth points')

    grid_size = config.grid_size
    grid_res  = 1 << config.min_lod

    z_min_val = float((depth_pts[:, 2].min() // grid_size) * grid_size - 5 * grid_size)

    # VOT
    device  = 'cuda'
    depth_t = th.from_numpy(depth_np).unsqueeze(0).to(device)   # (1, H, W)
    mask_t  = th.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).to(device)
    K_t     = th.from_numpy(K_np).unsqueeze(0).to(device)        # (1, 3, 3)
    z_min_t = th.tensor([[z_min_val]], device=device)             # (1, 1) → broadcast

    with th.no_grad():
        xy_ = th.linspace(0., grid_size * (grid_res - 1), grid_res, device=device) \
              - (grid_size * (grid_res - 1)) // 2
        z_  = xy_ + (grid_size * grid_res) // 2
        all_vc = th.stack(th.meshgrid(xy_, xy_, z_, indexing='ij'), dim=-1).reshape(-1, 3)
        all_vc = all_vc.unsqueeze(0)                    # (1, N, 3)
        all_vc[:, :, 2] += z_min_val

        K_norm     = K_t.clone()
        K_norm[:, :2] *= vot_inv
        depth_norm = -F.max_pool2d(-depth_t.unsqueeze(1), vot_sf).squeeze(1)
        mask_norm  = F.interpolate(mask_t.unsqueeze(1), scale_factor=vot_inv,
                                   mode='bilinear', align_corners=True).bool().squeeze(1)
        occ = vot(all_vc, mask_norm, depth_norm, K_norm)   # (1, N)

    occ_count = int(occ[0].sum().item())
    n_depth   = depth_pts.shape[0]
    mask_px   = int(mask_np.sum())
    d_min     = float(depth_pts[:, 2].min())
    d_max     = float(depth_pts[:, 2].max())
    return occ_count, n_depth, z_min_val, mask_px, d_min, d_max


def load_raw(tf, key):
    """Read raw fields from an open tarfile."""
    def read_json(name):
        return json.loads(tf.extractfile(f'{key}.{name}').read())

    def read_depth(name):
        data = tf.extractfile(f'{key}.{name}').read()
        arr  = np.asarray(Image.open(io.BytesIO(data)), dtype=np.float32)
        return arr

    return {
        'camera.json':     read_json('camera.json'),
        'depth.png':       read_depth('depth.png'),
        'gt_info.json':    read_json('gt_info.json'),
        'mask_visib.json': read_json('mask_visib.json'),
    }


def scan(config, out_dir):
    out_json = os.path.join(out_dir, 'empty_vot_samples.json')
    vot, vot_sf, vot_inv = build_vot(config)

    shard_paths = sorted(glob.glob(SHARD_PATTERN))
    print(f'Found {len(shard_paths)} shards')

    empty_samples = []
    total = 0

    for shard_idx, shard_path in enumerate(shard_paths):
        try:
            with tarfile.open(shard_path, 'r') as tf:
                keys = sorted(set(m.name.split('.')[0] for m in tf.getmembers()
                                  if m.name.endswith('.camera.json')))
                for key in keys:
                    try:
                        raw = load_raw(tf, key)
                        occ_count, n_depth, z_min_val, mask_px, d_min, d_max = \
                            process_raw_sample(raw, config, vot, vot_sf, vot_inv)
                        total += 1

                        if occ_count == 0:
                            entry = {
                                'frame_idx': key,
                                'shard':     shard_path,
                                'n_depth_pts': n_depth,
                                'mask_px':   mask_px,
                                'z_min':     z_min_val,
                                'depth_min': d_min,
                                'depth_max': d_max,
                            }
                            empty_samples.append(entry)
                            print(f'  [EMPTY] {key}  n_depth={n_depth}'
                                  f'  mask_px={mask_px}  z_min={z_min_val:.0f}'
                                  f'  depth={d_min:.0f}~{d_max:.0f}')
                    except Exception as e:
                        pass   # skip bad samples silently

        except Exception as e:
            print(f'[shard error] {shard_path}: {e}')
            continue

        if (shard_idx + 1) % 10 == 0:
            print(f'  shards {shard_idx+1}/{len(shard_paths)}'
                  f'  samples={total}  empty={len(empty_samples)}')

    print(f'\nDone. {len(empty_samples)} empty / {total} total')
    with open(out_json, 'w') as f:
        json.dump({'total': total, 'empty': empty_samples}, f, indent=2)
    print(f'Results → {out_json}')
    return empty_samples, out_json


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_entry(entry, config, out_dir):
    shard_path = entry.get('shard')
    key        = entry['frame_idx']

    if shard_path is None or not os.path.exists(shard_path):
        # try to find shard
        for sp in sorted(glob.glob(SHARD_PATTERN)):
            with tarfile.open(sp, 'r') as tf:
                names = {m.name.split('.')[0] for m in tf.getmembers()}
                if key in names:
                    shard_path = sp
                    break

    if shard_path is None:
        print(f'[warn] shard not found for {key}')
        return

    with tarfile.open(shard_path, 'r') as tf:
        raw = load_raw(tf, key)
        try:
            rgb = np.asarray(Image.open(io.BytesIO(
                tf.extractfile(f'{key}.rgb.jpg').read())))
        except Exception:
            rgb = None

    cam         = raw['camera.json']
    K_np        = np.asarray(cam['cam_K'], dtype=np.float32).reshape(3, 3)
    depth_np    = raw['depth.png'] * cam['depth_scale']
    gt_info     = raw['gt_info.json']
    H, W        = depth_np.shape
    vot_sf      = getattr(config, 'vot_scale_factor', 2)
    vot_inv     = 1.0 / vot_sf

    # Re-build mask
    mask_rle = raw['mask_visib.json']
    mask_np  = np.zeros((H, W), dtype=bool)
    for ind, oi in enumerate(gt_info):
        if oi.get('visib_fract', 0.0) < 0.2:
            continue
        rle = mask_rle.get(str(ind))
        if rle is not None:
            mask_np = np.logical_or(mask_np, rle_to_binary_mask(rle, oi.get('bbox_visib')))
    mask_depth = np.logical_and(mask_np, depth_np > 10.0)

    depth_ds = -F.max_pool2d(
        th.from_numpy(-depth_np).unsqueeze(0).unsqueeze(0).float(), vot_sf
    ).squeeze().numpy()
    mask_ds = F.interpolate(
        th.from_numpy(mask_depth.astype(np.float32)).unsqueeze(0).unsqueeze(0),
        scale_factor=vot_inv, mode='bilinear', align_corners=True
    ).bool().squeeze().numpy()

    visib_fracts = [round(oi.get('visib_fract', 0.0), 2) for oi in gt_info]

    ncols = 3 + (1 if rgb is not None else 0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f'{key}  [{os.path.basename(shard_path)}]\n'
        f'n_depth={entry["n_depth_pts"]}  mask_px={entry["mask_px"]}'
        f'  z_min={entry["z_min"]:.0f} mm\n'
        f'depth={entry["depth_min"]:.0f}~{entry["depth_max"]:.0f} mm'
        f'  visib_fracts={visib_fracts}',
        fontsize=9)

    r, c = 0, 0
    if rgb is not None:
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB')
        axes[0, 0].axis('off')
    else:
        axes[0, 0].axis('off')

    dm = axes[0, 1].imshow(depth_np, cmap='plasma')
    axes[0, 1].set_title(f'Depth (mm)')
    axes[0, 1].axis('off')
    fig.colorbar(dm, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(mask_depth, cmap='gray')
    axes[0, 2].set_title(f'Mask (depth>10) #{mask_depth.sum()}px')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(depth_ds, cmap='plasma')
    axes[1, 0].set_title(f'Depth /{vot_sf} (VOT input)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_ds, cmap='gray')
    axes[1, 1].set_title(f'Mask /{vot_sf} (VOT input)  #{mask_ds.sum()}px')
    axes[1, 1].axis('off')

    fg_depth = depth_np[mask_depth]
    if fg_depth.size > 0:
        axes[1, 2].hist(fg_depth, bins=50, color='steelblue')
        axes[1, 2].set_xlabel('depth (mm)')
        axes[1, 2].set_title('Foreground depth histogram')
        z_min = entry['z_min']
        grid_max = z_min + config.grid_size * (1 << config.min_lod)
        axes[1, 2].axvline(z_min,     color='red',    linestyle='--', label=f'z_min={z_min:.0f}')
        axes[1, 2].axvline(grid_max,  color='orange', linestyle='--', label=f'z_max={grid_max:.0f}')
        axes[1, 2].legend(fontsize=8)
    else:
        axes[1, 2].set_title('No foreground depth')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'{key}.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out_path}')


def visualize_all(empty_samples, config, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for entry in empty_samples:
        try:
            visualize_entry(entry, config, out_dir)
        except Exception as e:
            print(f'[vis error] {entry["frame_idx"]}: {e}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',  default='vot_empty_vis')
    parser.add_argument('--no_vis',   action='store_true')
    parser.add_argument('--vis_only', action='store_true')
    cli_args, _ = parser.parse_known_args()

    config = parse_config()
    config.update_octree = True

    os.makedirs(cli_args.out_dir, exist_ok=True)
    out_json = os.path.join(cli_args.out_dir, 'empty_vot_samples.json')

    if cli_args.vis_only:
        with open(out_json) as f:
            data = json.load(f)
        empty_samples = data['empty']
        print(f'Loaded {len(empty_samples)} empty samples from {out_json}')
    else:
        empty_samples, out_json = scan(config, cli_args.out_dir)

    if not cli_args.no_vis:
        vis_dir = os.path.join(cli_args.out_dir, 'vis')
        if empty_samples:
            print(f'\nVisualizing {len(empty_samples)} samples → {vis_dir}')
            visualize_all(empty_samples, config, vis_dir)
        else:
            print('No empty-VOT samples found.')


if __name__ == '__main__':
    main()
