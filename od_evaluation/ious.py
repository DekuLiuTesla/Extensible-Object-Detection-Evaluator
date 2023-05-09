import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from mmdet3d.core import LiDARInstance3DBoxes
except ImportError:
    LiDARInstance3DBoxes = None

try:
    from mmdet3d.core import bbox_overlaps_3d
except ImportError:
    bbox_overlaps_3d = None

def get_waymo_iou_matrix(preds, gts):

    if torch is None:
        raise ImportError('This function requires PyTorch.')
    if LiDARInstance3DBoxes is None or bbox_overlaps_3d is None:
        raise ImportError('This function requires mmdet3d.')

    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).cuda()
    if isinstance(gts, np.ndarray):
        gts = torch.from_numpy(gts).cuda()

    if preds.shape[0] == 0 or gts.shape[0] == 0:
        return np.zeros((preds.shape[0], gts.shape[0]), dtype=np.float32)
    assert preds.shape[0] > 0 and preds.shape[-1] == 7
    assert gts.shape[0] > 0 and gts.shape[-1] == 7

    preds = LiDARInstance3DBoxes(preds, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))
    gts = LiDARInstance3DBoxes(gts, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))

    ious = bbox_overlaps_3d(preds.tensor, gts.tensor, mode='iou', coordinate='lidar') #[num_preds, num_gts]
    assert ious.size(0) == preds.tensor.size(0)
    return ious.cpu().numpy()

def get_waymo_iou_matrix_bev(preds, gts):

    if torch is None:
        raise ImportError('This function requires PyTorch.')
    if LiDARInstance3DBoxes is None or bbox_overlaps_3d is None:
        raise ImportError('This function requires mmdet3d.')

    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds).cuda()
    if isinstance(gts, np.ndarray):
        gts = torch.from_numpy(gts).cuda()

    if preds.shape[0] == 0 or gts.shape[0] == 0:
        return np.zeros((preds.shape[0], gts.shape[0]), dtype=np.float32)
    assert preds.shape[0] > 0 and preds.shape[-1] == 7
    assert gts.shape[0] > 0 and gts.shape[-1] == 7

    preds = LiDARInstance3DBoxes(preds, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))
    gts = LiDARInstance3DBoxes(gts, box_dim=7, with_yaw=True, origin=(0.5, 0.5, 0.5))

    preds_bev = preds.tensor
    preds_bev[:, 2] = 0
    preds_bev[:, 5] = 2
    gts_bev = gts.tensor
    gts_bev[:, 2] = 0
    gts_bev[:, 5] = 2

    ious = bbox_overlaps_3d(preds_bev, gts_bev, mode='iou', coordinate='lidar') #[num_preds, num_gts]
    # print(preds_bev, gts_bev, ious)
    assert ious.size(0) == preds.tensor.size(0)
    return ious.cpu().numpy()