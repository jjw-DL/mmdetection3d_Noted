# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet.models import DETECTORS
from .mvx_two_stage import MVXTwoStageDetector


@DETECTORS.register_module()
class MVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN."""

    def __init__(self, **kwargs):
        super(MVXFasterRCNN, self).__init__(**kwargs)


@DETECTORS.register_module()
class DynamicMVXFasterRCNN(MVXTwoStageDetector):
    """Multi-modality VoxelNet using Faster R-CNN and dynamic voxelization."""

    def __init__(self, **kwargs):
        super(DynamicMVXFasterRCNN, self).__init__(**kwargs)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res) # （19221， 3)
            coors.append(res_coors)
        # 将点拼接
        points = torch.cat(points, dim=0)
        coors_batch = []
        # 添加batch信息
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        # 在batch维度拼接
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_pts_feat(self, points, img_feats, img_metas):
        """Extract point features."""
        if not self.with_pts_bbox:
            return None
        voxels, coors = self.voxelize(points) # (19221, 4),(19221, 4) 调用self.pts_voxel_layer计算每个点voxle坐标
        voxel_features, feature_coors = self.pts_voxel_encoder(
            voxels, coors, points, img_feats, img_metas) # (16160,128), (16160, 4)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size) # (1, 256, 200, 176)
        x = self.pts_backbone(x) # (1,128,200,176), (1,256,100,88)
        if self.with_pts_neck:
            x = self.pts_neck(x) # (1, 512, 200, 176)
        return x
