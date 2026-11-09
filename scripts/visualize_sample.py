"""
Visualize one sample from a shard tar: depth pts, SPC pts, tactile pts.
Saves PNG with 3 views (Front XZ, Side YZ, Top XY).

Usage:
  python scripts/visualize_sample.py \
      --shard /nas1/1000_Members/drag1008/TactMAE/train/shapenet_000000.tar \
      --out /tmp/sample_vis.png
"""

import argparse
import io
import json
import tarfile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_sample(tf, key):
    # depth → camera frame mm
    cam = json.loads(tf.extractfile(f"{key}.camera.json").read())
    K = np.array(cam['cam_K'], dtype=np.float32).reshape(3, 3)
    depth_scale = cam['depth_scale']
    depth = np.asarray(Image.open(io.BytesIO(tf.extractfile(f"{key}.depth.png").read())), dtype=np.float32) * depth_scale
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3).astype(np.float32)
    rays = (uv1 @ np.linalg.inv(K).T).reshape(H, W, 3)
    mask = (depth > 10.0) & (depth < 2000.0)
    depth_pts = (rays * depth[:, :, None])[mask].astype(np.float32)

    # SPC (already in camera frame mm)
    spc = np.load(io.BytesIO(tf.extractfile(f"{key}.spc.npz").read()))['spc'].astype(np.float32)

    # tactile (optional)
    try:
        tactile = np.load(io.BytesIO(tf.extractfile(f"{key}.tactile.npz").read()))['pts'].astype(np.float32)
    except KeyError:
        tactile = None

    return depth_pts, spc, tactile


def plot_views(depth_pts, spc, tactile, out_path, key):
    rng = np.random.default_rng(0)
    d = depth_pts[rng.choice(len(depth_pts), min(5000, len(depth_pts)), replace=False)]
    s = spc[rng.choice(len(spc), min(3000, len(spc)), replace=False)]
    t = tactile[rng.choice(len(tactile), min(1000, len(tactile)), replace=False)] if tactile is not None else None

    axes_pairs = [(0, 2, 'X', 'Z', 'Front (XZ)'),
                  (1, 2, 'Y', 'Z', 'Side (YZ)'),
                  (0, 1, 'X', 'Y', 'Top (XY)')]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(key, fontsize=9)

    for ax, (xi, yi, xl, yl, title) in zip(axes, axes_pairs):
        ax.scatter(d[:, xi], d[:, yi], s=1,  c='steelblue',  alpha=0.3, label=f'depth ({len(depth_pts)})')
        ax.scatter(s[:, xi], s[:, yi], s=1,  c='tomato',     alpha=0.5, label=f'SPC ({len(spc)})')
        if t is not None:
            ax.scatter(t[:, xi], t[:, yi], s=6, c='limegreen', alpha=0.9, label=f'tactile ({len(tactile)})')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel(xl + ' (mm)')
        ax.set_ylabel(yl + ' (mm)')
        if yi == 1:  # Y is down in camera frame
            ax.invert_yaxis()
        ax.legend(loc='upper right', markerscale=4, fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', required=True)
    parser.add_argument('--key',   default=None, help='sample key (default: first in shard)')
    parser.add_argument('--out',   default='/tmp/sample_vis.png')
    args = parser.parse_args()

    with tarfile.open(args.shard, 'r') as tf:
        keys = sorted(set(m.name.split('.')[0] for m in tf.getmembers()))
        key = args.key or keys[0]
        print(f"Key: {key}")
        depth_pts, spc, tactile = load_sample(tf, key)

    print(f"depth: {len(depth_pts)} pts  z={depth_pts[:,2].min():.1f}~{depth_pts[:,2].max():.1f} mm")
    print(f"SPC:   {len(spc)} pts  z={spc[:,2].min():.1f}~{spc[:,2].max():.1f} mm  sdf={spc[:,3].min():.3f}~{spc[:,3].max():.3f}")
    if tactile is not None:
        print(f"tactile: {len(tactile)} pts  z={tactile[:,2].min():.1f}~{tactile[:,2].max():.1f} mm")
    else:
        print("tactile: none")

    plot_views(depth_pts, spc, tactile, args.out, key)


if __name__ == '__main__':
    main()
