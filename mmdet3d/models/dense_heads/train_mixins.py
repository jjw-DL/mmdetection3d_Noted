# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmdet3d.core import limit_period
from mmdet.core import images_to_levels, multi_apply


class AnchorTrainMixin(object):
    """Mixin class for target assigning of dense heads."""

    def anchor_target_3d(self,
                         anchor_list,
                         gt_bboxes_list,
                         input_metas,
                         gt_bboxes_ignore_list=None,
                         gt_labels_list=None,
                         label_channels=1,
                         num_classes=1,
                         sampling=True):
        """Compute regression and classification targets for anchors.

        Args:
            anchor_list (list[list]): Multi level anchors of each image.
            gt_bboxes_list (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                bboxes of each image.
            input_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (None | list): Ignore list of gt bboxes.
            gt_labels_list (list[torch.Tensor]): Gt labels of batches.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple (list, list, list, list, list, list, int, int):
                Anchor targets, including labels, label weights,
                bbox targets, bbox weights, direction targets,
                direction weights, number of postive anchors and
                number of negative anchors.
        """
        num_imgs = len(input_metas) # 2
        assert len(anchor_list) == num_imgs

        if isinstance(anchor_list[0][0], list):
            # sizes of anchors are different
            # anchor number of a single level
            num_level_anchors = [
                sum([anchor.size(0) for anchor in anchors])
                for anchors in anchor_list[0]
            ]
            for i in range(num_imgs):
                anchor_list[i] = anchor_list[i][0]
        else:
            # anchor number of multi levels
            num_level_anchors = [
                anchors.view(-1, self.box_code_size).size(0)
                for anchors in anchor_list[0]
            ] # 211200

            # concat all level anchors and flags to a single tensor
            for i in range(num_imgs):
                anchor_list[i] = torch.cat(anchor_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        
        # ????????????
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         all_dir_targets, all_dir_weights, pos_inds_list,
         neg_inds_list) = multi_apply(
             self.anchor_target_3d_single,
             anchor_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             input_metas,
             label_channels=label_channels,
             num_classes=num_classes,
             sampling=sampling)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list]) # 129 + 126 = 255
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list]) # 210857 + 210849 = 421706
        # split targets to a list w.r.t. multiple levels
        # SECOND???????????????????????????????????????????????????list????????????????????????
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        dir_targets_list = images_to_levels(all_dir_targets, num_level_anchors)
        dir_weights_list = images_to_levels(all_dir_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, dir_targets_list, dir_weights_list,
                num_total_pos, num_total_neg)

    def anchor_target_3d_single(self,
                                anchors,
                                gt_bboxes,
                                gt_bboxes_ignore,
                                gt_labels,
                                input_meta,
                                label_channels=1,
                                num_classes=1,
                                sampling=True):
        """Compute targets of anchors in single batch.

        Args:
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            label_channels (int): The channel of labels.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        # self.bbox_assigner???MaxIouAssinger
        if isinstance(self.bbox_assigner,
                      list) and (not isinstance(anchors, list)):
            # anchors --> (1, 200, 176, 3, 2, 7)
            feat_size = anchors.size(0) * anchors.size(1) * anchors.size(2) # 35200
            rot_angles = anchors.size(-2) # 2
            assert len(self.bbox_assigner) == anchors.size(-3)
            # ?????????
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            # ??????3??????????????????assign
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[..., i, :, :].reshape(
                    -1, self.box_code_size) #???70400, 7???
                current_anchor_num += current_anchors.size(0) # ????????????anchor?????????
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes[gt_per_cls, :],
                        gt_bboxes_ignore, gt_labels[gt_per_cls], input_meta,
                        num_classes, sampling)
                else:
                    # ??????????????????????????????????????????for????????????????????????size???????????????????????????????????????
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes, gt_bboxes_ignore,
                        gt_labels, input_meta, num_classes, sampling)
                
                # ???assign??????????????????
                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, pos_inds, neg_inds) = anchor_targets
                
                total_labels.append(labels.reshape(feat_size, 1, rot_angles)) # (35200, 1, 2)
                total_label_weights.append(
                    label_weights.reshape(feat_size, 1, rot_angles))
                
                total_bbox_targets.append(
                    bbox_targets.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1))) # (35200, 1, 2, 7)
                total_bbox_weights.append(
                    bbox_weights.reshape(feat_size, 1, rot_angles,
                                         anchors.size(-1)))
                
                total_dir_targets.append(
                    dir_targets.reshape(feat_size, 1, rot_angles))
                total_dir_weights.append(
                    dir_weights.reshape(feat_size, 1, rot_angles))
                
                total_pos_inds.append(pos_inds) # 28,...
                total_neg_inds.append(neg_inds) # 70318,...

            # ???num_size???????????? --> 211200
            total_labels = torch.cat(total_labels, dim=-2).reshape(-1) # ???211200, ???
            total_label_weights = torch.cat(
                total_label_weights, dim=-2).reshape(-1) # ???211200, ???
            total_bbox_targets = torch.cat(
                total_bbox_targets, dim=-3).reshape(-1, anchors.size(-1)) # ???211200???7???
            total_bbox_weights = torch.cat(
                total_bbox_weights, dim=-3).reshape(-1, anchors.size(-1)) # ???211200???7???
            total_dir_targets = torch.cat(
                total_dir_targets, dim=-2).reshape(-1) # ???211200, ???
            total_dir_weights = torch.cat(
                total_dir_weights, dim=-2).reshape(-1) # ???211200, ???
            total_pos_inds = torch.cat(total_pos_inds, dim=0).reshape(-1) # (129, )
            total_neg_inds = torch.cat(total_neg_inds, dim=0).reshape(-1) # (210857,)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        elif isinstance(self.bbox_assigner, list) and isinstance(
                anchors, list):
            # class-aware anchors with different feature map sizes
            assert len(self.bbox_assigner) == len(anchors), \
                'The number of bbox assigners and anchors should be the same.'
            (total_labels, total_label_weights, total_bbox_targets,
             total_bbox_weights, total_dir_targets, total_dir_weights,
             total_pos_inds, total_neg_inds) = [], [], [], [], [], [], [], []
            current_anchor_num = 0
            for i, assigner in enumerate(self.bbox_assigner):
                current_anchors = anchors[i]
                current_anchor_num += current_anchors.size(0)
                if self.assign_per_class:
                    gt_per_cls = (gt_labels == i)
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes[gt_per_cls, :],
                        gt_bboxes_ignore, gt_labels[gt_per_cls], input_meta,
                        num_classes, sampling)
                else:
                    anchor_targets = self.anchor_target_single_assigner(
                        assigner, current_anchors, gt_bboxes, gt_bboxes_ignore,
                        gt_labels, input_meta, num_classes, sampling)

                (labels, label_weights, bbox_targets, bbox_weights,
                 dir_targets, dir_weights, pos_inds, neg_inds) = anchor_targets
                total_labels.append(labels)
                total_label_weights.append(label_weights)
                total_bbox_targets.append(
                    bbox_targets.reshape(-1, anchors[i].size(-1)))
                total_bbox_weights.append(
                    bbox_weights.reshape(-1, anchors[i].size(-1)))
                total_dir_targets.append(dir_targets)
                total_dir_weights.append(dir_weights)
                total_pos_inds.append(pos_inds)
                total_neg_inds.append(neg_inds)

            total_labels = torch.cat(total_labels, dim=0)
            total_label_weights = torch.cat(total_label_weights, dim=0)
            total_bbox_targets = torch.cat(total_bbox_targets, dim=0)
            total_bbox_weights = torch.cat(total_bbox_weights, dim=0)
            total_dir_targets = torch.cat(total_dir_targets, dim=0)
            total_dir_weights = torch.cat(total_dir_weights, dim=0)
            total_pos_inds = torch.cat(total_pos_inds, dim=0)
            total_neg_inds = torch.cat(total_neg_inds, dim=0)
            return (total_labels, total_label_weights, total_bbox_targets,
                    total_bbox_weights, total_dir_targets, total_dir_weights,
                    total_pos_inds, total_neg_inds)
        else:
            return self.anchor_target_single_assigner(self.bbox_assigner,
                                                      anchors, gt_bboxes,
                                                      gt_bboxes_ignore,
                                                      gt_labels, input_meta,
                                                      num_classes, sampling)

    def anchor_target_single_assigner(self,
                                      bbox_assigner,
                                      anchors,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      input_meta,
                                      num_classes=1,
                                      sampling=True):
        """Assign anchors and encode positive anchors.

        Args:
            bbox_assigner (BaseAssigner): assign positive and negative boxes.
            anchors (torch.Tensor): Concatenated multi-level anchor.
            gt_bboxes (:obj:`BaseInstance3DBoxes`): Gt bboxes.
            gt_bboxes_ignore (torch.Tensor): Ignored gt bboxes.
            gt_labels (torch.Tensor): Gt class labels.
            input_meta (dict): Meta info of each image.
            num_classes (int): The number of classes.
            sampling (bool): Whether to sample anchors.

        Returns:
            tuple[torch.Tensor]: Anchor targets.
        """
        anchors = anchors.reshape(-1, anchors.size(-1)) # ???70400, 7???
        num_valid_anchors = anchors.shape[0] # 70400
        # ?????????
        bbox_targets = torch.zeros_like(anchors) # ???70400, 7???
        bbox_weights = torch.zeros_like(anchors) # ???70400, 7???
        dir_targets = anchors.new_zeros((anchors.shape[0]), dtype=torch.long) #???70400,???
        dir_weights = anchors.new_zeros((anchors.shape[0]), dtype=torch.float) #???70400,???
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long) #???70400,???
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float) #???70400,???
        if len(gt_bboxes) > 0:
            if not isinstance(gt_bboxes, torch.Tensor):
                gt_bboxes = gt_bboxes.tensor.to(anchors.device)
            # AssignResult(num_gts=23, gt_inds.shape=(70400,), max_overlaps.shape=(70400,), labels.shape=(70400,))
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
            # PseudoSampler: Directly returns the positive and negative indices  of samples.
            # ??????SamplingResult???
            sampling_result = self.bbox_sampler.sample(assign_result, anchors,
                                                       gt_bboxes)
            pos_inds = sampling_result.pos_inds
            neg_inds = sampling_result.neg_inds
        else:
            pos_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) > 0,
                as_tuple=False).squeeze(-1).unique()
            neg_inds = torch.nonzero(
                anchors.new_zeros((anchors.shape[0], ), dtype=torch.bool) == 0,
                as_tuple=False).squeeze(-1).unique()

        if gt_labels is not None:
            labels += num_classes
        if len(pos_inds) > 0:
            # ?????????box?????????????????????box targets
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            # ????????????box????????????bin
            pos_dir_targets = get_direction_target(
                sampling_result.pos_bboxes,
                pos_bbox_targets,
                self.dir_offset,
                one_hot=False)
            # ?????????bbox???target??????bbox_targets,??????loss??????
            bbox_targets[pos_inds, :] = pos_bbox_targets  # (70400, 7)
            bbox_weights[pos_inds, :] = 1.0
            dir_targets[pos_inds] = pos_dir_targets
            dir_weights[pos_inds] = 1.0

            # ???lable????????????????????????????????????????????????
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds] # (70400)
            # ????????????????????????
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        # ????????????????????????
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # ?????????????????????anchors??????????????? 70400
        return (labels, label_weights, bbox_targets, bbox_weights, dir_targets,
                dir_weights, pos_inds, neg_inds)


def get_direction_target(anchors,
                         reg_targets,
                         dir_offset=0,
                         num_bins=2,
                         one_hot=True):
    """Encode direction to 0 ~ num_bins-1.

    Args:
        anchors (torch.Tensor): Concatenated multi-level anchor.
        reg_targets (torch.Tensor): Bbox regression targets.
        dir_offset (int): Direction offset.
        num_bins (int): Number of bins to divide 2*PI.
        one_hot (bool): Whether to encode as one hot.

    Returns:
        torch.Tensor: Encoded direction targets.
    """
    rot_gt = reg_targets[..., 6] + anchors[..., 6] # ?????????+anchor
    # ??????val???[-offset * period, (1-offset) * period]??????
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi) # ????????????0???2pi??????
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long() # ???????????????bin?????????bin????????????2???????????????????????????0???1
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1) # ???bin??????clamp
    # ??????one-hot??????
    if one_hot:
        dir_targets = torch.zeros(
            *list(dir_cls_targets.shape),
            num_bins,
            dtype=anchors.dtype,
            device=dir_cls_targets.device)
        dir_targets.scatter_(dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
        dir_cls_targets = dir_targets
    return dir_cls_targets
