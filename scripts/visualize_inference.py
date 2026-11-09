"""
Inference + multi-view visualization for a single sample.

Usage:
  python scripts/visualize_inference.py \
      --config configs/shapenet_tactile_scratch.yaml \
      --checkpoint checkpoints/octmae_shapenet/<run>/epoch=X-step=Y.ckpt \
      --frame_idx 02876657_10f709cecfbb8d59c2536abb1e8e5eab_00000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch as th
import open3d as o3d
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import webdataset as wds

from train import BaseTrainer
from octmae.utils.config import parse_config
from octmae.utils.misc import (
    make_sample_wrapper, decode_depth,
    unnormalize_pts, project_to_image_plane, batch2cuda
)
from octmae.nets.utils import get_xyz_from_octree

INTRINSICS_K = {
    'shapenet': [[572.0, 0., 320.0], [0., 572.0, 240.0], [0., 0., 1.]],
}
TEST_URL = {
    'shapenet': '/nas1/1000_Members/drag1008/TactMAE/test/shapenet_{000000..000022}.tar',
}


def render_pcd_multiview(pcd_o3d, width=480, height=480):
    """Render Open3D point cloud from 4 views. Returns list of HxWx3 numpy images."""
    mat = rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    mat.point_size = 2.5

    center = np.asarray(pcd_o3d.get_center())
    extent = np.asarray(pcd_o3d.get_axis_aligned_bounding_box().get_extent())
    dist = float(np.max(extent)) * 2.2

    # (name, eye_offset, up)  — all relative to center
    views = [
        ('Front',     np.array([0,  0, -dist]),         np.array([0, -1, 0])),
        ('Side',      np.array([dist, 0,  0]),           np.array([0, -1, 0])),
        ('Top',       np.array([0, -dist, 0]),           np.array([0,  0,  1])),
        ('Isometric', np.array([dist*0.7, -dist*0.7, -dist*0.7]), np.array([0, -1, 0])),
    ]

    images = []
    for name, offset, up in views:
        r = rendering.OffscreenRenderer(width, height)
        r.scene.set_background([0.12, 0.12, 0.12, 1.0])
        r.scene.add_geometry('pcd', pcd_o3d, mat)
        eye = center + offset
        r.scene.camera.look_at(center.tolist(), eye.tolist(), up.tolist())
        img = np.asarray(r.render_to_image())
        images.append((name, img))
        r.scene.clear_geometry()

    return images


def find_sample(config, frame_idx, dataset_name='shapenet'):
    url = TEST_URL[dataset_name]
    K = INTRINSICS_K[dataset_name]
    config_eval = argparse.Namespace(**vars(config))
    config_eval.update_octree = True

    wrapper = make_sample_wrapper(config_eval, K=K, is_eval=True)

    dataset = (
        wds.WebDataset(url, nodesplitter=wds.split_by_node,
                       handler=wds.warn_and_continue, shardshuffle=False)
        .decode(decode_depth, 'pil', handler=wds.warn_and_continue)
        .map(wrapper, handler=wds.warn_and_continue)
    )

    for sample in dataset:
        if sample[-1] == frame_idx:
            return sample
    return None


def make_gt_pcd(sample, config):
    grid_res = 1 << config.min_lod
    pts_3d_gt = sample[5]
    z_min = sample[7]
    unnorm = unnormalize_pts(pts_3d_gt.points.clone(), z_min, config.grid_size, grid_res)
    normals = pts_3d_gt.normals
    colors = (normals + 1.0) / 2.0
    gt_o3d = o3d.geometry.PointCloud()
    gt_o3d.points = o3d.utility.Vector3dVector(unnorm.cpu().numpy())
    gt_o3d.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
    return gt_o3d


def run_inference(model, sample, config):
    grid_res = 1 << config.min_lod
    batch = [s.unsqueeze(0) if th.is_tensor(s) else [s] for s in sample]
    batch = batch2cuda(batch)
    z_min = batch[7][0]

    with th.no_grad():
        output = model.model(batch)
        octrees_out = output['octrees_out']
        pcd = get_xyz_from_octree(octrees_out, config.max_lod, True)
        pcd = unnormalize_pts(pcd, z_min, config.grid_size, grid_res)
        normals = octrees_out.normals[config.max_lod]
        sdf = octrees_out.features[config.max_lod][:, :1]

        pcd = pcd.cpu().numpy()
        normals_np = normals.cpu().numpy()
        sdf_np = sdf.cpu().numpy()
        pcd = pcd - normals_np * sdf_np

    pred_o3d = o3d.geometry.PointCloud()
    pred_o3d.points = o3d.utility.Vector3dVector(pcd)
    colors = (normals_np + 1.0) / 2.0
    pred_o3d.colors = o3d.utility.Vector3dVector(colors)
    return pred_o3d


def save_multiview_figure(frame_idx, gt_o3d, pred_o3d, rgb_np, output_path):
    gt_views = render_pcd_multiview(gt_o3d)
    pred_views = render_pcd_multiview(pred_o3d)
    view_names = [v[0] for v in gt_views]

    fig = plt.figure(figsize=(22, 12))
    fig.suptitle(frame_idx, fontsize=11)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.08, wspace=0.04)

    # Row 0: RGB image (span first 2 cols) + blank
    ax_rgb = fig.add_subplot(gs[0, :2])
    ax_rgb.imshow(rgb_np)
    ax_rgb.set_title('RGB Input', fontsize=9)
    ax_rgb.axis('off')

    # Row 1: GT views
    for col, (name, img) in enumerate(gt_views):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img)
        ax.set_title(f'GT – {name}', fontsize=9)
        ax.axis('off')

    # Row 2: Prediction views
    for col, (name, img) in enumerate(pred_views):
        ax = fig.add_subplot(gs[2, col])
        ax.imshow(img)
        ax.set_title(f'Pred – {name}', fontsize=9)
        ax.axis('off')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_idx', type=str, required=True,
                        help='e.g. 02876657_10f709cecfbb8d59c2536abb1e8e5eab_00000')
    parser.add_argument('--output_dir', type=str, default='vis_results')
    cli_args, _ = parser.parse_known_args()

    config = parse_config()
    config.update_octree = True

    os.makedirs(cli_args.output_dir, exist_ok=True)
    output_path = os.path.join(cli_args.output_dir, f'{cli_args.frame_idx}.png')

    print(f'Searching for sample: {cli_args.frame_idx}')
    sample = find_sample(config, cli_args.frame_idx)
    if sample is None:
        print('Sample not found.')
        return

    print('Loading model...')
    model = BaseTrainer.load_from_checkpoint(config.checkpoint, config=config)
    model.cuda()
    model.eval()

    print('Running inference...')
    pred_o3d = run_inference(model, sample, config)
    gt_o3d = make_gt_pcd(sample, config)

    # RGB for display (denormalize)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std  = np.array([0.26862954, 0.26130258, 0.27577711])
    rgb_np = sample[0].permute(1, 2, 0).cpu().numpy()
    rgb_np = np.clip(rgb_np * std + mean, 0, 1)

    print('Rendering multi-view...')
    save_multiview_figure(cli_args.frame_idx, gt_o3d, pred_o3d, rgb_np, output_path)


if __name__ == '__main__':
    main()
