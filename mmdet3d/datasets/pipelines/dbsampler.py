# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv
import numpy as np
import os

from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.pipelines import data_augment_utils
from mmdet.datasets import PIPELINES
from ..builder import OBJECTSAMPLERS


class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str | None): The category of samples. Default: None.
        epoch (int | None): Sampling epoch. Default: None.
        shuffle (bool): Whether to shuffle indices. Default: False.
        drop_reminder (bool): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices) # 打乱索引
        self._idx = 0
        self._example_num = len(sampled_list) # gt的数量
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num: # 采样数量如果大于gt的数量
            ret = self._indices[self._idx:].copy() # 赋值全部索引作为结果
            self._reset() # 将idx重置为0
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0 # 将idx重置为0

    def sample(self, num):
        """Sample specific number of ground truths.
        采样一定数量的gt

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices] # info信息组成的list


@OBJECTSAMPLERS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3])):
        super().__init__()
        self.data_root = data_root # database的存储路径
        self.info_path = info_path # infos文件的路径
        self.rate = rate # 1.0
        self.prepare = prepare # 点云预处理pipeline
        self.classes = classes # 采样类别
        self.cat2label = {name: i for i, name in enumerate(classes)} # 下面两行将类别和数字对应
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES) # 初始化点云加载器

        db_infos = mmcv.load(info_path) # 加载infos文件

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos') # 输出database的各种类别gt的数量 eg:load 65262 truck database infos
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val) # 对db_infos进行预处理
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos') # 输出database的各种类别gt的数量

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)}) # 将类别和数量字典加入sample_groups

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys()) # 将类别组建列表
            self.sample_max_nums += list(group_info.values()) # 将对应类别数量组建列表

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True) # 针对类别进行采样，传入sampled_list和类别,返回值实际是一个Class
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        # 逐个类别进行过滤
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        # 逐个类别按照最少点数进行过滤
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                # 逐类别过滤
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos # 将过滤后的infos重新写入
        return db_infos   

    def sample_all(self, gt_bboxes, gt_labels, img=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        # 按照类别和采样个数(cfg文件给出)，逐类计算采样个数
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name] # 获取类别对应的数字
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels])) # 最大采样数-当前gt里面该类别的个数设置为当前采样数
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64) # 按照比率重新计算采样数
            sampled_num_dict[class_name] = sampled_num # 该类别的采样数目,可能为负数
            sample_num_per_class.append(sampled_num) # 每个类别的采样数量

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes # 要避免碰撞的box eg: (32,9)

        # 逐类按照按照采样数进行采样
        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            # 如果要采样数大于0
            if sampled_num > 0:
                # 采样该类别一定数量的box的info
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls # 记录所有采样box的info，不分类别
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...] # 如果只有一个box，则需要在最前面增加一个维度，方便后面concatenate
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0) # 将采样的box进行拼接
                    # list相加为拼接，这里是list的元素为list，每个list为不同类别的采样gt box
                    sampled_gt_bboxes += [sampled_gt_box] 
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0) # 将采样的box和原始box进行拼接，更新要避免碰撞的box
                    # tip:concatenate和stack的区别是concatenate不会新曾维度，stack会
        
        # 对result进行整合，包括box拼接，点云获取和label获取
        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0) # 将所有采样的gt box进行拼接，忽略类别因素
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = [] # 初始化点云列表
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path'] # 拼接文件路径
                results = dict(pts_filename=file_path) # 构造results字典
                s_points = self.points_loader(results)['points'] # 根据路径加载点云
                s_points.translate(info['box3d_lidar'][:3]) # 因为在生成gt点云的时候，减去了box的偏移，这里要加回来将点云移入box内

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long) # 构造label
            # 返回的是采样的伪gt，不包括原始gt
            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list), # 对采样点云进行拼接，因为点云是无顺序的，直接拼接即可
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled)) # 在原始box的基础上增加id
            }

        return ret

    def sample_class_v2(self, name, num, gt_bboxes):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num) # 采样num个box的info
        sampled = copy.deepcopy(sampled) # 进行深拷贝
        num_gt = gt_bboxes.shape[0] # 获取gt box的数量 eg:32
        num_sampled = len(sampled) # 获取采样数量 eg:4
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]) # 将gt box3D转换到bev视角 eg:(32,4,2)

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0) # 将sample info中的采样box进行堆叠 eg:(4,9)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy() # 将gt和采样box堆叠 eg:(36,9)
        # 获取采样的box-->这里为什么不直接取sp_boxes而是先拼接所有box，然后再截取呢？感觉多此一举
        sp_boxes_new = boxes[gt_bboxes.shape[0]:] # eg:(4,9)
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]) # 将采样的box转换到bev视角 eg:(4,4,2)

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0) # 在bev视角将gt和sp box进行拼接 eg:(36,4,2)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv) # 检查box之间是否存在碰撞 eg:(36,36)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False # box自己一定不会和自己相撞

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled): # 只对采样box进行处理
            # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True
            # 也就是说如果第i个采样box与任何一个box存在碰撞,则将该box包行的行和列全部设置为False
            if coll_mat[i].any(): 
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt]) # 否则在有效采样box中添加该info
        return valid_samples # 返回有效采样的info
