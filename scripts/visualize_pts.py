"""
Batch visualize .pts files saved by eval.py, optionally alongside GT.

Usage (pred only):
  python scripts/visualize_pts.py --input eval_results/.../meshes/

Usage (pred + GT, dataset streamed once):
  python scripts/visualize_pts.py \
      --input eval_results/.../meshes/ \
      --config configs/shapenet_no_tactile_scratch.yaml \
      --dataset_name shapenet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

import argparse
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import webdataset as wds
from octmae.utils.config import parse_config
from octmae.utils.misc import (
    make_sample_wrapper, decode_depth,
    unnormalize_pts, project_to_image_plane,
)

TEST_URL = {
    'shapenet': '/nas1/1000_Members/drag1008/TactMAE/test/shapenet_{000000..000022}.tar',
}
INTRINSICS_K = {
    'shapenet': [[572.0, 0., 320.0], [0., 572.0, 240.0], [0., 0., 1.]],
}

# (name, elev, azim) — camera-frame: X right, Y down, Z into scene
VIEWS = [
    ('Front',     -10,  -90),
    ('Side',      -10,    0),
    ('Top',        90,  -90),
    ('Isometric',  20,  -60),
]


def compute_bounds(*pcds):
    """Return shared axis limits from combined point clouds."""
    all_pts = np.concatenate([np.asarray(p.points) for p in pcds], axis=0)
    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
    center = (lo + hi) / 2
    half = (hi - lo).max() / 2 * 1.1
    return center, half


def render_scatter(pts, colors, bounds, px=480):
    """Render 4 views of a point cloud as numpy images using matplotlib scatter."""
    center, half = bounds
    imgs = []
    for name, elev, azim in VIEWS:
        fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor((0.12, 0.12, 0.12))
        ax.set_facecolor((0.12, 0.12, 0.12))

        # subsample for speed if very dense
        N = pts.shape[0]
        if N > 20000:
            idx = np.random.choice(N, 20000, replace=False)
            p = pts[idx]
        else:
            p = pts

        import matplotlib.cm as cm
        z = p[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        c = cm.plasma(z_norm)[:, :3]

        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c=c, s=5, linewidths=0, depthshade=True)

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, color='white', fontsize=9, pad=2)
        ax.set_axis_off()
        plt.tight_layout(pad=0)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        imgs.append((name, img))
        plt.close(fig)
    return imgs


def save_figure(frame_idx, pred_o3d, output_path, gt_o3d=None):
    pred_pts = np.asarray(pred_o3d.points)
    pred_col = np.asarray(pred_o3d.colors) if pred_o3d.has_colors() else None

    bounds = compute_bounds(pred_o3d, gt_o3d) if gt_o3d is not None else compute_bounds(pred_o3d)

    n_rows = 2 if gt_o3d is not None else 1
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    fig.suptitle(frame_idx, fontsize=10)
    if n_rows == 1:
        axes = [axes]

    for col, (name, img) in enumerate(render_scatter(pred_pts, pred_col, bounds)):
        axes[0][col].imshow(img)
        axes[0][col].set_title(f'Pred – {name}', fontsize=9)
        axes[0][col].axis('off')

    if gt_o3d is not None:
        gt_pts = np.asarray(gt_o3d.points)
        gt_col = np.asarray(gt_o3d.colors) if gt_o3d.has_colors() else None
        for col, (name, img) in enumerate(render_scatter(gt_pts, gt_col, bounds)):
            axes[1][col].imshow(img)
            axes[1][col].set_title(f'GT – {name}', fontsize=9)
            axes[1][col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {output_path}')


def make_gt_o3d(sample, config):
    import torch as th
    grid_res = 1 << config.min_lod
    pts_3d_gt = sample[5]
    z_min = sample[7]
    K = sample[6]

    unnorm = unnormalize_pts(pts_3d_gt.points.clone(), z_min, config.grid_size, grid_res)
    pts_2d = project_to_image_plane(unnorm.clone(), K, (config.img_height, config.img_width))
    mask = th.all(pts_2d > -1.0, dim=1) & th.all(pts_2d < 1.0, dim=1)
    unnorm = unnorm[mask].cpu().numpy()
    normals = pts_3d_gt.normals[mask].cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(unnorm)
    pcd.colors = o3d.utility.Vector3dVector((normals + 1.0) / 2.0)
    return pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--max_files', type=int, default=100)
    parser.add_argument('--per_category', type=int, default=None,
                        help='If set, sample this many files per ShapeNet category')
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip', dest='skip_existing', action='store_false')
    parser.add_argument('--config', default=None)
    parser.add_argument('--dataset_name', default='shapenet')
    args, remaining = parser.parse_known_args()

    if os.path.isfile(args.input):
        pts_files = [args.input]
        default_out = os.path.dirname(args.input)
    else:
        pts_files = sorted(glob.glob(os.path.join(args.input, '*.pts')))
        pts_files = [p for p in pts_files if not p.endswith('_gt.pts')]
        default_out = args.input

    if not pts_files:
        print('No .pts files found.')
        return

    if args.per_category is not None:
        from collections import defaultdict
        cat_map = defaultdict(list)
        for p in pts_files:
            cat = os.path.basename(p).split('_')[0]
            cat_map[cat].append(p)
        pts_files = []
        for cat, files in sorted(cat_map.items()):
            pts_files.extend(files[:args.per_category])
        print(f'Categories: {sorted(cat_map.keys())}')
    else:
        pts_files = pts_files[:args.max_files]
    output_dir = args.output_dir or default_out
    os.makedirs(output_dir, exist_ok=True)

    pred_map = {}
    target_ids = set()
    for pts_path in pts_files:
        frame_idx = os.path.splitext(os.path.basename(pts_path))[0]
        out_path = os.path.join(output_dir, f'{frame_idx}.png')
        if args.skip_existing and os.path.exists(out_path):
            print(f'  skip: {frame_idx}')
            continue
        pcd = o3d.io.read_point_cloud(pts_path)
        if len(pcd.points) == 0:
            print(f'  [warn] empty: {pts_path}')
            continue
        pred_map[frame_idx] = pcd
        target_ids.add(frame_idx)

    if not pred_map:
        print('Nothing to render.')
        return

    gt_map = {}
    if args.config is not None:
        sys.argv = [sys.argv[0], '--config', args.config] + remaining
        config = parse_config()
        config.update_octree = True

        url = TEST_URL[args.dataset_name]
        K = INTRINSICS_K[args.dataset_name]
        wrapper = make_sample_wrapper(config, K=K, is_eval=True)

        n_total = len(target_ids)
        print(f'Streaming dataset for GT ({n_total} frames)...')
        dataset = (
            wds.WebDataset(url, nodesplitter=wds.split_by_node,
                           handler=wds.warn_and_continue, shardshuffle=False)
            .decode(decode_depth, 'pil', handler=wds.warn_and_continue)
            .map(wrapper, handler=wds.warn_and_continue)
        )
        for i, sample in enumerate(dataset):
            fid = sample[-1]
            if fid in target_ids:
                gt_map[fid] = make_gt_o3d(sample, config)
                target_ids.discard(fid)
                print(f'  GT found {n_total - len(target_ids)}/{n_total}: {fid}', flush=True)
            if not target_ids:
                break
            if i % 500 == 0 and i > 0:
                print(f'  ... scanned {i} samples, {len(target_ids)} remaining', flush=True)

    total = len(pred_map)
    print(f'Rendering {total} file(s) → {output_dir}')
    for i, (frame_idx, pred_o3d) in enumerate(pred_map.items(), 1):
        print(f'[{i}/{total}] {frame_idx} ...', flush=True)
        out_path = os.path.join(output_dir, f'{frame_idx}.png')
        save_figure(frame_idx, pred_o3d, out_path, gt_map.get(frame_idx))


if __name__ == '__main__':
    main()
