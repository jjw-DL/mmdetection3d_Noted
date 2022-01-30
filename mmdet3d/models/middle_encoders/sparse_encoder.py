# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from ..builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module'):
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape # [41, 1600, 1408]
        self.in_channels = in_channels # 4
        self.order = order # ('conv', 'norm', 'act')
        self.base_channels = base_channels # 16
        self.output_channels = output_channels # 128
        self.encoder_channels = encoder_channels # ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64))
        self.encoder_paddings = encoder_paddings # ((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1))
        self.stage_num = len(self.encoder_channels) # 4
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        # 初始Stage:voxle encoder进入3D卷积的第一阶段
        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate 初始化
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        # 中间stage:[1600, 1408, 41] -> [800, 704, 21] -> [400, 352, 11] -> [200, 176, 5]
        #                   16       ->       32       ->      64       ->      64
        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type) # ‘basicblock’

        # 最后stage:[200, 176, 5] -> [200, 176, 2]
        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        # 对输入voxle特征进行稀疏编码，转化为稀疏tensor
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor) # 对输入的稀疏tensor进行编码，16（41, 1600, 1408）

        encode_features = []
        for encoder_layer in self.encoder_layers:
            #（2, 16, 41, 1600, 1408）
            # (2, 32, 41, 800, 704)
            # (2, 64, 11, 400, 352)
            # (2, 64, 5, 200, 176)
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [2, 64, 5, 200, 176] -> [2, 128, 2, 200, 176]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape # 2, 128, 2, 200, 176
        spatial_features = spatial_features.view(N, C * D, H, W) # (2, 256, 200, 176)

        return spatial_features

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                # 如果该block是conv_module，不是第一个block，是第一层则进入该判断
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                # 如果该block是basicblock
                elif block_type == 'basicblock':
                    # 是该block的最后一层，且不是最后一个block
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        # 如果该block是basicblock且不满足上面的条件则进入该判断
                        # Sparse basic block implemented with submanifold sparse convolution
                        # 继承自resnet的BasicBlock和spconv的SparseModule
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels # 更新输入通道
            stage_name = f'encoder_layer{i + 1}' # 更新层名
            stage_layers = spconv.SparseSequential(*blocks_list) # 序列化该层
            self.encoder_layers.add_module(stage_name, stage_layers) # 将该layer添加进encoder_layers
        return out_channels
