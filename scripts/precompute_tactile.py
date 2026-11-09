"""
Precompute per-view tactile point clouds in camera frame (mm).

For each rendered view in ShapeNetCoreV2Rendered, loads all 20 tactile touches
for the corresponding object and transforms them into the view's camera frame.

Coordinate chain:
  sensor local  →  PyBullet world  →  original mesh frame  →  Blender world  →  camera (mm)

Usage:
  python scripts/precompute_tactile.py \
      --rendered_root /nas1/1000_Members/drag1008/ShapeNetCoreV2Rendered \
      --tactile_root /nas1/1000_Members/drag1008/TouchSDF/results/touch_charts_gt_ShapeNetCore_Original_Practice2 \
      --shapenet_root /nas1/1000_Members/drag1008/ShapeNetCore \
      --categories 02876657 02880940 02942699 03467517 03593526 03797390
"""

import argparse
import numpy as np
import torch
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# PyBullet simulation constants (from extract_touch_charts.yaml / script)
SCALE_PB   = 0.2
OBJ_POS    = np.array([0.5, 0.0, 0.0])
OBJ_RPY    = [np.pi / 2, 0.0, -np.pi / 2]
R_OBJ      = Rotation.from_euler('xyz', OBJ_RPY).as_matrix()   # mesh → PyBullet world
R_OBJ_INV  = R_OBJ.T                                            # PyBullet world → mesh
R_YUP      = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)  # OBJ Y-up → Blender Z-up


def pybullet_world_to_camera_mm(p_world: np.ndarray,
                                 bb_center: np.ndarray,
                                 bb_extent_max: float,
                                 w2c: np.ndarray) -> np.ndarray:
    """
    p_world:       (N, 3) points in PyBullet world frame (meters)
    bb_center:     (3,)   bounding box center of model_normalized.obj
    bb_extent_max: scalar bounding box max extent of model_normalized.obj
    w2c:           (4, 4) world-to-camera matrix (inv of cam2world.pt)
    returns:       (N, 3) in camera frame, mm
    """
    # PyBullet world → original mesh frame
    p_mesh = (R_OBJ_INV @ ((p_world - OBJ_POS) / SCALE_PB).T).T   # (N, 3)

    # original mesh → Blender rendered world (Y-up → Z-up, then normalize)
    p_zup    = (R_YUP @ p_mesh.T).T
    scale_bl = 0.2 / bb_extent_max
    p_blender = (p_zup - bb_center) * scale_bl                      # (N, 3)

    # Blender world → camera frame (meters)
    p_cam = (w2c[:3, :3] @ p_blender.T).T + w2c[:3, 3]             # (N, 3)

    return p_cam * 1000.0                                            # mm


def load_touches(touch_obj_dir: Path):
    """Return list of (p_local, R_tcp, t_tcp) for all available touch files."""
    touches = []
    for i in range(20):
        fpath = touch_obj_dir / f"{i}.npy"
        if not fpath.exists():
            continue
        d = np.load(fpath, allow_pickle=True).item()
        p_local = d['pointclouds'][0].astype(np.float64)      # (200, 3)
        R_tcp   = d['rot_M_wrld_list'][0].astype(np.float64)  # (3, 3)
        t_tcp   = d['pos_wrld_list'][0].astype(np.float64)    # (3,)
        touches.append((p_local, R_tcp, t_tcp))
    return touches


def touches_to_pybullet_world(touches):
    """Reconstruct world frame points from all touch samples → (N_total, 3)."""
    pts = []
    for p_local, R_tcp, t_tcp in touches:
        p_world = (R_tcp @ p_local.T).T + t_tcp   # (200, 3)
        pts.append(p_world)
    return np.concatenate(pts, axis=0)             # (20*200, 3)


def process_object(category_id: str, object_id: str,
                   rendered_root: Path, tactile_root: Path, shapenet_root: Path,
                   skip_existing: bool = True) -> int:
    touch_obj_dir = tactile_root / category_id / object_id
    if not touch_obj_dir.exists():
        return 0

    touches = load_touches(touch_obj_dir)
    if not touches:
        return 0

    # Mesh bounding box (same mesh Blender uses)
    mesh_path = shapenet_root / category_id / object_id / "models" / "model_normalized.obj"
    if not mesh_path.exists():
        return 0
    mesh = trimesh.load(str(mesh_path), force='mesh', process=False)
    verts_zup = (R_YUP @ mesh.vertices.T).T               # Y-up → Z-up
    bb_min, bb_max = verts_zup.min(axis=0), verts_zup.max(axis=0)
    bb_center      = (bb_min + bb_max) / 2.0
    bb_extent_max  = (bb_max - bb_min).max()

    # All tactile points in PyBullet world frame
    p_world_all = touches_to_pybullet_world(touches)      # (N, 3)

    cameras_dir = rendered_root / category_id / object_id / "cameras"
    if not cameras_dir.exists():
        return 0

    count = 0
    for view_dir in sorted(cameras_dir.iterdir()):
        out_path = view_dir / "tactile.npz"
        if skip_existing and out_path.exists():
            count += 1
            continue

        c2w_path = view_dir / "cam2world.pt"
        if not c2w_path.exists():
            continue

        c2w = torch.load(str(c2w_path), weights_only=True).numpy().astype(np.float64)
        w2c = np.linalg.inv(c2w)

        p_cam_mm = pybullet_world_to_camera_mm(p_world_all, bb_center, bb_extent_max, w2c)
        np.savez(str(out_path), pts=p_cam_mm.astype(np.float32))
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rendered_root', required=True)
    parser.add_argument('--tactile_root',  required=True)
    parser.add_argument('--shapenet_root', required=True)
    parser.add_argument('--categories', nargs='*', default=None,
                        help='process only these category IDs (default: all)')
    parser.add_argument('--no_skip_existing', action='store_true')
    args = parser.parse_args()

    rendered_root = Path(args.rendered_root)
    tactile_root  = Path(args.tactile_root)
    shapenet_root = Path(args.shapenet_root)
    skip_existing = not args.no_skip_existing

    if args.categories:
        categories = args.categories
    else:
        categories = sorted(p.name for p in rendered_root.iterdir() if p.is_dir())

    for cat_id in categories:
        cat_dir = rendered_root / cat_id
        if not cat_dir.is_dir():
            continue
        obj_ids = sorted(p.name for p in cat_dir.iterdir() if p.is_dir())
        for obj_id in tqdm(obj_ids, desc=cat_id):
            process_object(cat_id, obj_id, rendered_root, tactile_root,
                           shapenet_root, skip_existing)


if __name__ == '__main__':
    main()
