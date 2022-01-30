# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

# python tools/train.py ${CONFIG_FILE} [optional arguments]

# =========== optional arguments ===========
# --work-dir        存储日志和模型的目录
# --resume-from     加载 checkpoint 的目录
# --no-validate     是否在训练的时候进行验证
# 互斥组：
#   --gpus          使用的 GPU 数量
#   --gpu_ids       使用指定 GPU 的 id
# --seed            随机数种子
# --deterministic   是否设置 cudnn 为确定性行为
# --options         其他参数
# --launcher        分布式训练使用的启动器，可以为：['none', 'pytorch', 'slurm', 'mpi']
#                   none：不启动分布式训练，dist_train.sh 中默认使用 pytorch 启动。
# --local_rank      本地进程编号，此参数 torch.distributed.launch 会自动传入。

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path') # 配置文件路径
    parser.add_argument('--work-dir', help='the dir to save logs and models') # 输出文件路径
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from') # 断点续训练权重文件路径
    # action: store (默认, 表示保存参数)
    # action: store_true, store_false (如果指定参数, 则为 True, False)
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training') # 是否有validate阶段
    
    # 创建一个互斥组: argparse将会确保互斥组中只有一个参数在命令行中可用
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)') # 指定GPU数量
    # 可以使用 python train.py --gpu-ids 0 1 2 3 指定使用的 GPU id
    # 参数结果：[0, 1, 2, 3]
    # nargs = '*'：参数个数可以设置0个或n个
    # nargs = '+'：参数个数可以设置1个或n个
    # nargs = '?'：参数个数可以设置0个或1个
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)') # 指定GPU的id

    parser.add_argument('--seed', type=int, default=0, help='random seed') # 指定随机种子
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.') # 指定CUDNN backend是否为deterministic
    
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.') # 覆盖config中的配置
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    # 分布式训练相关参数
    # 如果使用 dist_utils.sh 进行分布式训练, launcher 默认为 pytorch
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    # 本地进程编号，此参数 torch.distributed.launch 会自动传入
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    # 解析命令行参数
    args = parser.parse_args() 

    # 如果环境中没有 LOCAL_RANK，就设置它为当前的 local_rank
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # options相关
    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    #----------------------------------------1.解析命令行参数和配置文件----------------------------------------#
    args = parse_args()
    # 从文件读取配置
    cfg = Config.fromfile(args.config)
    # 从命令行读取额外的配置
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options) # 将命令行配置合并到配置文件中
    # import modules from string list. 导入自定义模块
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    #----------------------------------------2.设置GPU和分布式相关参数----------------------------------------#
    # set cudnn_benchmark https://zhuanlan.zhihu.com/p/73711222
    """
    设置torch.backends.cudnn.benchmark=True将会让程序在开始时花费一点额外时间，
    为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速
    适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括batch size，图片大小，输入的通道）是不变的，
    其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间
    """
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # 设置GPU的id
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    # 如果 launcher 为 none，不启用分布式训练。不使用 dist_train.sh 默认参数为 none.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        # 初始化 dist 里面会调用 init_process_group
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    #----------------------------------------3.设置work_dir相关参数----------------------------------------#
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
     # 当 work_dir 为 None 的时候, 使用 ./work_dir/配置文件名 作为默认工作目录
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0]) # 配合文件去除后缀
    # create work_dir 创建输出文件夹
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config 将配置文件写入输出文件夹
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    #----------------------------------------4.设置log相关参数----------------------------------------#
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime()) # 获取时间戳
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log') # 拼接log文件路径
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    # 创建root logger
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    #----------------------------------------5.提取meta信息并写入log----------------------------------------#
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info 搜集环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()]) # 拼接环境信息
    dash_line = '-' * 60 + '\n' # 60个-的虚线
    # 记录基本环境信息
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info # 将环境信息写入meta字典
    meta['config'] = cfg.pretty_text # 将配置信息转变为易读的形式并写入meta字典

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    #----------------------------------------6.设置学习率和断点续训练------------------------------#
    # 设置学习率
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # 设置断点续训练
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    
    #----------------------------------------7.构建网络模型----------------------------------------#
    # 构建网络模型
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')

    #----------------------------------------8.构建数据集----------------------------------------#
    # 构建数据集: 需要传入 cfg.data.train
    datasets = [build_dataset(cfg.data.train)]

    #----------------------------------------9.设置验证集和checkpoint相关参数----------------------#
    # workflow 代表流程：
    # [('train', 2), ('val', 1)] 就代表，训练两个 epoch 验证一个 epoch
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)

    #----------------------------------------10.模型训练----------------------------------------#    
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES #  ['Pedestrian', 'Cyclist', 'Car']
    # 训练检测器, 传入：模型, 数据集, config 等
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
