# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from .builder import DATASETS


@DATASETS.register_module()
class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset # 数据集
        self.CLASSES = dataset.CLASSES # 类别
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)} # 类别到数字的映射
        self.sample_indices = self._get_sample_indices() # 计算采样索引
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()} # 以类别数字为key初始化字典 {1：{}, 2:{}, ...}
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx) # 获取该scene下的所有采样类别(数字)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx) # 在对应的类别id加入scene的id，记录每个类别包含的scene
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()]) # 计算所有类别采样总数
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        } # 计算数据集类别分布

        sample_indices = []

        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()] # 计算采样比率，数量少的类的采样比率会变大

        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist() # 计算采样索引（一个list）
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)
