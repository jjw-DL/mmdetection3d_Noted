# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True) # 利用官方API构造Dataset class
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train # 获取train对应的场景 700段
        val_scenes = splits.val # 获取val对应的场景 150段
    elif version == 'v1.0-test':
        train_scenes = splits.test # 获取test对应的场景 150段
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc) # 获取有效场景list
    available_scene_names = [s['name'] for s in available_scenes] # 将有效关键帧的名字组成list --> ['scene-0001', 'scene-0002',..., 'scene-1110']
    # 将train_scenes中有效scene组成train_scenes_names
    # map方法返回的新数组是原数组的映射，和原数组的长度相同,
    # filter方法返回的值是过滤原数组后的新数组，和原数组长度不同
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes)) 
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    # 将train_scenes中有效scene组成train_scenes_tokens
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ]) # 700
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ]) # 150

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    # 调用函数获取train和val的infos信息
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version) # v1.0-trainval
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        # 将train_nusc_infos和metadata组成字典，并写入pkl文件
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        # 针对 val 同理
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token'] # 获取scenes的token
        scene_rec = nusc.get('scene', scene_token) # 根据token获取scene的record，rec代表record
        sample_rec = nusc.get('sample', scene_rec['first_sample_token']) # 获取该scene下第一个sample的record
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP']) # 获取该sample下的Lidar Data的record
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:# 这里只判断第一sample是否存在，如果存在这该scene有效，否则无效
            # 返回data_path, box_list, cam_intrinsic
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token']) # 根据sample data的token获取标注信息
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1] # 获取相对路径
                # relative path
            # 判断lidar point是否存在
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True 
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene) # 在有效scene list中加入该场景信息
    print('exist scene num: {}'.format(len(available_scenes))) # 850:所有sample均有效
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset. --> Database Class
        train_scenes (list[str]): Basic information of training scenes. --> 700个token的list
        val_scenes (list[str]): Basic information of validation scenes. --> 150个token的list
        test (bool): Whether use the test mode. In the test mode, no --> False
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10. --> 10

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP'] # 获取lidar的token
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP']) # 获取sample_data中LIDAR_TOP的record
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token']) # 获取calibrated_sensor的record
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token']) # 获取自车pose的record
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token) # 获取lidar路径，标注的box和相机内参

        mmcv.check_file_exist(lidar_path) # 检测lidar文件是否存在

        info = {
            'lidar_path': lidar_path, # 点云文件的路径
            'token': sample['token'], # sampe的token
            'sweeps': [],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'], # lidar到自车的平移
            'lidar2ego_rotation': cs_record['rotation'], # lidar到自车的旋转，四元数形式
            'ego2global_translation': pose_record['translation'], # 自车的全局的平移
            'ego2global_rotation': pose_record['rotation'], # 自车的全局的旋转
            'timestamp': sample['timestamp'], # 时间戳
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix # 将四元数转化为旋转矩阵
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        # 逐相机处理
        for cam in camera_types:
            cam_token = sample['data'][cam] # sample['data'] 包含当前sample的全部传感器token，这里取出cam的token
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token) # 根据相机的token获取图片的路径和内参
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam) # 获取camera到lidar的变换矩阵R和t
            cam_info.update(cam_intrinsic=cam_intrinsic) # 在cam_info中加入intrinsic信息
            info['cams'].update({cam: cam_info}) # 将cam信息加入info中

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP']) # 获取当前sample_data的LIDAR_TOP record
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '': # 如果该sample的前面有数据，则获取前一帧的sweep信息
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep) # 加入sweep的list中
                sd_rec = nusc.get('sample_data', sd_rec['prev']) # 更新当前recore为前一帧的record
            else:
                break
        info['sweeps'] = sweeps

        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ] # 获取该sample中的所有sample_annotation的record-->一个sample_annotation对应一个Box实例 eg:10个
            locs = np.array([b.center for b in boxes]).reshape(-1, 3) # 位置 (10, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3) # 长宽高 (10, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1) # 旋转 (10, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']]) # 速度 (10, 2) 根据距离/时间估计，可能为nan
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1) # 点的个数大于0则标记为valid
            
            # convert velocity from global to lidar
            # 将速度从global系转换到lidar系
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes] # 组合box的类别list
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]] # 根据dataset中的映射关系，将当前类进行映射去掉属性值，共10类
            names = np.array(names) # （10, 1）
            # 重新构造gt box， we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1) # （10, 7）
            assert len(gt_boxes) == len(
                annotations), f'{len(gt_boxes)}, {len(annotations)}'
            # 将上面计算得到的值进行赋值，构造info信息
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = names
            info['gt_velocity'] = velocity.reshape(-1, 2)
            info['num_lidar_pts'] = np.array(
                [a['num_lidar_pts'] for a in annotations])
            info['num_radar_pts'] = np.array(
                [a['num_radar_pts'] for a in annotations])
            info['valid_flag'] = valid_flag

        # 如果该scene的token在train的场景中，则将info加入train_nusc_infos，否则加入val_nusc_infos
        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    # 返回train和val的info
    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
       获取camera到lidar的变换矩阵
    Args:
        nusc (class): Dataset class in the nuScenes dataset. --> Dataset Class
        sensor_token (str): Sample data token corresponding to the --> sensor_token
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token) # 根据sample内的sensor token获取sample_data record
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token']) # 根据calibrated_sensor_token获取calibrated_sensor record
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token']) # 根据ego_pose_token获取自车的pose
    data_path = str(nusc.get_sample_data_path(sd_rec['token'])) # 获取当前图片的路径
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    # 存储sweep的infos
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'], # sensor到自车的平移
        'sensor2ego_rotation': cs_record['rotation'], # sensor到自车的旋转quaternion
        'ego2global_translation': pose_record['translation'], # 自车到全局的平移
        'ego2global_rotation': pose_record['rotation'], # 自车到平移的旋转quaternion
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix # sensor到自车的旋转矩阵
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix # 自车到全局的旋转矩阵
    # 这里的计算过程经过推导没有问题，这里t(1x3)和R的转置是右乘关系-->t*R^T
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    # sensor到lidar的旋转和平移矩阵
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T

    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos'] # 加载之前生成的pkl文件的infos信息
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True) # 构造Dataset Class
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ] # ['id':0, 'name':'car', ...] 
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids) # 初始化coco_2d_dict
    # 逐个标注处理
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam] # 从之前生成的info文件中获取相机的info
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d) # 获取coco格式的infos
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape # 获取图像宽高
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d' # nuscenes_infos_train_mono3d
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json') # nuscenes_infos_train_mono3d.coco.json


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter. --> ['', '1', '2', '3', '4']
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token) # 根据sample_data的token获取sample_data的record

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token']) # 根据sample_token获取sample的record

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token']) # 根据sample_data的token获取calibrated_sensor的record
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token']) # # 根据sample_data的token获取ego_pose的record
    camera_intrinsic = np.array(cs_rec['camera_intrinsic']) # 根据calibrated_sensor的record获取标定信息

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ] # 获取该sample下的所有annotation
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ] # 根据可见性过滤掉一些annotation

    repro_recs = [] # 初始化

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information. 在sample_annotation中增加字段
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token']) # 调用API获取当前ann_rec下的Box实例（在全局坐标系下）

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse) # 四元数初始化并取逆

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten() # 深度要大于0
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords # 获取box的左上和右下角坐标

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            # 前面已经将box从global系变换到cam系了，因此这里不用变化
            loc = box.center.tolist() # 获取box的中心坐标 3维

            dim = box.wlh # 获取box的长宽高
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0] # 获取box的方向
            rot = [-rot]  # convert the rot to our cam coordinate
            # 将速度转换到cam系下，global-->ego-->cam
            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True) # 将中心点从cam系变换到img系
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens'] # 获取sample_annotation record中的attribute_tokens
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name'] # 获取attribute record中的name
            attr_id = nus_attributes.index(attr_name) # 根据name获取索引
            repro_rec['attribute_name'] = attr_name # 添加attribute_name字段
            repro_rec['attribute_id'] = attr_id # 添加attribute_id属性

        repro_recs.append(repro_rec) # 在repro_recs添加repro_rec

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas) # 计算box3d投影与图片的交集坐标
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])
        # 获取吐包的外接矩形的左上角和右下角坐标
        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    # 初始化
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    # 相关键值
    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    # 如果ann_rec中的key存在于relevant_keys中直接赋值
    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2] # 添加box的corner信息
    repro_rec['filename'] = filename # 添加文件名
    # coco格式
    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    # 如果类别不再10类里面则直接返回None
    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    # 获取对应的映射类别
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name) # 获取类别索引
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1] # 将box坐标更改为左上和宽高形式
    coco_rec['iscrowd'] = 0
    # 为什么不返回repro_rec？？？如果只是想生成coco格式record，前面repro_rec的计算和赋值是为了什么？
    return coco_rec
