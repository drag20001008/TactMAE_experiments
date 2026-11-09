import os
import json

from scipy.spatial import KDTree
import numpy as np
import open3d as o3d
import torch as th
import webdataset as wds

from train_latent_lod import BaseTrainer
from octmae.utils.config import parse_config
from octmae.utils.misc import (
    make_sample_wrapper,
    decode_depth,
    unnormalize_pts,
    project_to_image_plane,
    batch2cuda
)
from octmae.nets.utils import get_xyz_from_octree


EVAL_DATA_SET_NAMES = ['synth_eval', 'ycb_video', 'hope', 'hb', 'shapenet']
IMG_SIZES = {
    'synth_eval': (480, 640),
    'ycb_video': (480, 640),
    'hope': (1080, 1920),
    'hb': (480, 640),
    'shapenet': (480, 640),
}
INTRINSICS_K = {
    'synth_eval': [[572.41136339, 0., 325.2611084], [0., 573.57043286, 242.04899588], [0., 0., 1.]],
    'ycb_video': [[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]],
    'hope': [[1390.53, 0.0, 964.957], [0.0, 1386.99, 522.586], [0.0, 0.0, 1.0]],
    'hb': [[537.4799, 0.0, 318.8965], [0.0, 536.1447, 238.3781], [0.0, 0.0, 1.0]],
    'shapenet': [[572.0, 0., 320.0], [0., 572.0, 240.0], [0., 0., 1.]],
}

S3_URL = 'your_s3_path'
WDS_URL = {
    'synth_eval': os.path.join(S3_URL, 'eval', 'synth_eval', 'shard-{000000..000009}.tar'),
    'ycb_video': os.path.join(S3_URL, 'eval', 'ycb_video', 'shard-{000000..000009}.tar'),
    'hope': os.path.join(S3_URL, 'eval', 'hope', 'shard-000000.tar'),
    'hb': os.path.join(S3_URL, 'eval', 'hb', 'shard-{000000..000009}.tar'),
    'shapenet': '/nas1/1000_Members/drag1008/TactMAE/test/shapenet_{000000..000022}.tar',
}

def evaluate():
    config = parse_config()
    config_name = config.eval_name if config.eval_name else os.path.basename(config.config).replace('.yaml', '')
    dataset_name = config.eval_dataset_name
    if dataset_name not in EVAL_DATA_SET_NAMES:
        raise Exception('this dataset is not supported')

    # update config
    img_size = IMG_SIZES[dataset_name]
    config.img_height = img_size[0]
    config.img_width = img_size[1]
    config.update_octree = True

    result_dir = os.path.join('eval_results', config.model_name, config_name, dataset_name)
    mesh_dir = os.path.join(result_dir, 'meshes')
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir, exist_ok=True)

    print('Loading a model...')
    model = BaseTrainer.load_from_checkpoint(config.checkpoint, config=config, strict=False)
    model.cuda()
    model.eval()

    print('Fetching data...')
    raw_url = WDS_URL[dataset_name]
    url = f'pipe:s5cmd cat {raw_url}' if raw_url.startswith('s3://') else raw_url
    dataset = (
        wds.WebDataset(url, nodesplitter=wds.split_by_node, handler=wds.warn_and_continue, shardshuffle=False)
        .decode(decode_depth, 'pil')
        .map(make_sample_wrapper(config, K=INTRINSICS_K[dataset_name], is_eval=True), handler=wds.warn_and_continue)
        .batched(1)
    )
    dataloader = wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=True
    )

    results = predict_surface(model, dataloader, mesh_dir, config)

    print(results)

    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(results, f)


def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None,
                        thresholds=np.array([5.0, 10.0, 20.0, 30.0])):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
            thresholds (numpy array): threshold values for the F-score calculation
        '''

        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        recall = get_threshold_percentage(completeness, thresholds)
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = completeness_normals.mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        precision = get_threshold_percentage(accuracy, thresholds)
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = accuracy_normals.mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)

        # F-Score
        F = [
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            for i in range(len(precision))
        ]

        out_dict = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'f-score-5': F[0], # threshold = 5mm
            'f-score-10': F[1], # threshold = 10mm
            'f-score-20': F[2], # threshold = 20mm
            'f-score-30': F[3], # threshold = 30mm
        }

        return out_dict


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold


def split_visible_occluded(unnorm_pts, pts_2d, depth, occlusion_threshold=10.0):
    """Split GT points into visible/occluded using depth map comparison.

    A point is visible if its Z is within occlusion_threshold mm of the depth map
    at its projected pixel. Otherwise it is occluded (something closer blocks it).

    Args:
        unnorm_pts: (N, 3) tensor, camera-frame mm
        pts_2d: (N, 2) tensor, normalized image coords (-1, 1)
        depth: (H, W) tensor, depth map in mm
        occlusion_threshold: mm threshold to classify visible vs occluded

    Returns:
        visible_mask, occluded_mask: boolean tensors of shape (N,)
    """
    H, W = depth.shape
    # Convert (-1,1) → pixel index
    px = ((pts_2d[:, 0] + 1.0) / 2.0 * W).long().clamp(0, W - 1)
    py = ((pts_2d[:, 1] + 1.0) / 2.0 * H).long().clamp(0, H - 1)

    depth_at_proj = depth[py, px]           # depth map value at projected pixel
    pt_depth = unnorm_pts[:, 2]             # point's actual Z in camera frame (mm)
    depth_diff = pt_depth - depth_at_proj   # positive = point is behind visible surface

    visible_mask = depth_diff < occlusion_threshold
    occluded_mask = ~visible_mask
    return visible_mask, occluded_mask


def _metric_keys(suffix=''):
    keys = [
        'completeness', 'accuracy', 'normals completeness', 'normals accuracy',
        'normals', 'completeness2', 'accuracy2', 'chamfer-L2', 'chamfer-L1',
        'f-score-5', 'f-score-10', 'f-score-20', 'f-score-30',
    ]
    return {f'{k}{suffix}': [] for k in keys}


def predict_surface(model, dataloader, mesh_dir, config):
    grid_res = 1 << config.min_lod

    out_dict = {'frame_idxs': [], 'runtime_reg': []}
    out_dict.update(_metric_keys(''))           # overall
    out_dict.update(_metric_keys('-visible'))   # visible region
    out_dict.update(_metric_keys('-occluded'))  # occluded region

    for batch in dataloader:
        metrics = {}
        batch = batch2cuda(batch)

        frame_idx = batch[-1][0]
        K = batch[-3][0]
        z_min = batch[-2][0]
        pts_3d_gt = batch[5][0]
        depth = batch[2][0]     # (H, W) depth map in mm
        pcd_path = os.path.join(mesh_dir, f'{frame_idx}.pts')

        if os.path.exists(pcd_path):
            pcd_ = o3d.io.read_point_cloud(pcd_path)
            pcd = np.asarray(pcd_.points)
            normals = np.asarray(pcd_.colors) * 2 - 1.0
        else:
            print(f'Running inference for {frame_idx}')
            with th.no_grad():
                output = model.model(batch)

                octrees_out = output['octrees_out']
                pcd = get_xyz_from_octree(octrees_out, config.max_lod, True)
                pcd = unnormalize_pts(pcd, z_min, config.grid_size, grid_res)
                normals = octrees_out.normals[config.max_lod]
                sdf = octrees_out.features[config.max_lod][:, :1]

                pcd = pcd.cpu().numpy()
                normals = normals.cpu().numpy()
                sdf = sdf.cpu().numpy()
                pcd = pcd - normals * sdf

            pcd_sdf_vis = o3d.geometry.PointCloud()
            pcd_sdf_vis.points = o3d.utility.Vector3dVector(pcd)
            if normals is not None:
                pcd_sdf_vis.normals = o3d.utility.Vector3dVector(normals)
                pcd_sdf_vis.colors = o3d.utility.Vector3dVector(((normals + 1) / 2))
            else:
                pcd_sdf_vis.estimate_normals()
                normals = np.asarray(pcd_sdf_vis.normals)
            o3d.io.write_point_cloud(pcd_path, pcd_sdf_vis)
            print('pcd is exported to', pcd_path)

        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        # GT points visible in image frustum
        unnorm_pts_3d_gt = unnormalize_pts(pts_3d_gt.points.clone(), z_min, config.grid_size, grid_res)
        pts_2d_gt = project_to_image_plane(unnorm_pts_3d_gt.clone(), K, (config.img_height, config.img_width))
        pts_mask_min = th.all(pts_2d_gt > -1.0, dim=1)
        pts_mask_max = th.all(pts_2d_gt < 1.0, dim=1)
        frustum_mask = th.logical_and(pts_mask_min, pts_mask_max)

        unnorm_pts_frustum = unnorm_pts_3d_gt[frustum_mask]
        pts_2d_frustum = pts_2d_gt[frustum_mask]
        gt_normals_frustum = pts_3d_gt.normals[frustum_mask]

        # Overall
        gt_pcd = unnorm_pts_frustum.cpu().numpy()
        gt_normals = gt_normals_frustum.cpu().numpy()
        metric = eval_pointcloud(pcd, gt_pcd, normals, gt_normals)
        metrics.update({k: v for k, v in metric.items()})

        # Visible / occluded split
        vis_mask, occ_mask = split_visible_occluded(unnorm_pts_frustum, pts_2d_frustum, depth)

        # Split pred points by the same visibility criterion
        pcd_tensor = th.tensor(pcd, dtype=th.float32).to(depth.device)
        pred_2d = project_to_image_plane(pcd_tensor.clone(), K, (config.img_height, config.img_width))
        pred_vis_mask, pred_occ_mask = split_visible_occluded(pcd_tensor, pred_2d, depth)

        for suffix, gt_mask, pred_mask in [
            ('-visible',  vis_mask,  pred_vis_mask),
            ('-occluded', occ_mask,  pred_occ_mask),
        ]:
            sub_gt  = gt_pcd[gt_mask.cpu().numpy()]
            sub_gt_nrm = gt_normals[gt_mask.cpu().numpy()]
            sub_pcd = pcd[pred_mask.cpu().numpy()]
            sub_nrm = normals[pred_mask.cpu().numpy()]
            if sub_gt.shape[0] > 0 and sub_pcd.shape[0] > 0:
                sub_metric = eval_pointcloud(sub_pcd, sub_gt, sub_nrm, sub_gt_nrm)
            else:
                sub_metric = {k: float('nan') for k in metric.keys()}
            metrics.update({f'{k}{suffix}': v for k, v in sub_metric.items()})

        print(frame_idx, {k: v for k, v in metrics.items() if 'chamfer-L1' in k or 'f-score-10' in k})

        for k in metrics.keys():
            if k in out_dict:
                out_dict[k].append(metrics[k])
        out_dict['frame_idxs'].append(frame_idx)

    for k in out_dict.keys():
        if k == 'frame_idxs':
            continue
        vals = [v for v in out_dict[k] if not np.isnan(v)]
        out_dict[k] = np.mean(np.asarray(vals)).item() if vals else float('nan')

    return out_dict


if __name__ == '__main__':
    evaluate()
