#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import glob
import os
import sys
from PIL import Image
from typing import NamedTuple
import pdb

import cv2
from tqdm import tqdm
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    primx:float
    primy:float
    depth_params: dict
    image_path: str
    mask_path: str
    depth_path: str
    image_name: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.quantile(dist, 0.9)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readCamerasFromCityTransforms(path, transformsfile, extension=".png", is_debug=False, undistorted=False,scale=0.1, add_depth=False):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None

        frames = contents["frames"]
        # check if filename already contain postfix
        if frames[0]["file_path"].split('.')[-1] in ['jpg', 'jpeg', 'JPG', 'png']:
            extension = ""

        c2ws = np.array([frame["transform_matrix"] for frame in frames])
        
        Ts = c2ws[:,:3,3]

        ct = 0

        progress_bar = tqdm(frames, desc="Loading city dataset")
        depth_params = {"scale":scale, "depth_scale":(5 / scale)}

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            if "matrix" in cam_name:
                cam_name = cam_name.replace("pjlab-lingjun-landmarks","pjlab_lingjun_landmarks")
            if not os.path.exists(cam_name):
                continue
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])

            if idx % 10 == 0:
                progress_bar.set_postfix({"num": Fore.YELLOW+f"{ct}/{len(frames)}"+Style.RESET_ALL})
                progress_bar.update(10)
            if idx == len(frames) - 1:
                progress_bar.close()
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            c2w[:3,3]/=scale
            
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # import pdb;pdb.set_trace()
            if undistorted:
                mtx = np.array(
                    [
                        [frame["fl_x"], 0, frame["cx"]],
                        [0, frame["fl_y"], frame["cy"]],
                        [0, 0, 1.0],
                    ],
                    dtype=np.float32,
                )
                dist = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"], frame["k3"]], dtype=np.float32)
                im_data = np.array(image.convert("RGB"))
                arr = cv2.undistort(im_data / 255.0, mtx, dist, None, mtx)
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])
            
            mask_path = ""
            depth_path = ""
            if add_depth:
                depth_path = frame["depth_path"]
            primx = 0.5 # assume simple pinhole
            primy = 0.5
            is_test = (transformsfile == 'transforms_test.json')
            
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, primx=primx, primy=primy, depth_params=depth_params,
                              image_path=image_path, mask_path=mask_path, depth_path=depth_path, image_name=image_name, 
                              width=image.size[0], height=image.size[1], is_test=is_test)
            
            cam_infos.append(cam_info)
            
            # cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            #                 image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
            if is_debug and idx > 50:
                break
    return cam_infos

def fetchPly(path, ratio=1):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if('red' in vertices):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.ones_like(positions) * 0.5
    if('nx' in vertices):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions[::ratio], colors=colors[::ratio], normals=normals[::ratio])

def fetchPt(xyz_path, rgb_path):
    positions_tensor = torch.jit.load(xyz_path).state_dict()['0']

    positions = positions_tensor.numpy()

    colors_tensor = torch.jit.load(rgb_path).state_dict()['0']
    if colors_tensor.size(0) == 0:
        colors_tensor = 255 * (torch.ones_like(positions_tensor) * 0.5)
    colors = (colors_tensor.float().numpy()) / 255.0
    normals = torch.Tensor([]).numpy()

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCityInfo(path, eval,ratio=1,scale=0.1, llffhold=50, type='merge'):
    # load ply
    ply_path = glob.glob(os.path.join(path, "*.ply"))[0]
    if os.path.exists(ply_path):
        try:
            pcd = fetchPly(ply_path,ratio=ratio)
        except:
            raise ValueError("must have tiepoints!")
    
    pcd.points[:,:] /=scale
    
    extension = 'png'
    is_debug = False
    
    is_chunk = False
    if "transforms_train.json" not in os.listdir(path):
        is_chunk = True
        
    if is_chunk:
        train_cam_infos = readCamerasFromCityTransforms(path, "transforms.json", extension, is_debug=is_debug,scale=scale)
        test_cam_infos = []
        
        if eval:
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx % llffhold == 0]
    else:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromCityTransforms(path, "transforms_train.json", extension, is_debug=is_debug,scale=scale, add_depth=True)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromCityTransforms(path, "transforms_test.json", extension, is_debug=is_debug,scale=scale, add_depth=False)
    
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []
    
    if type != "merge":
        if type == 'aerial':
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if "street" not in c.image_path]
            test_cam_infos = [c for idx, c in enumerate(test_cam_infos) if "street" not in c.image_path]
        elif type == 'street':
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if "street" in c.image_path]
            test_cam_infos = [c for idx, c in enumerate(test_cam_infos) if "street" in c.image_path]

    print(len(test_cam_infos), "test images")
    print(len(train_cam_infos), "train images")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info