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

import math
import os
import torch
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render, render_post
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
import numpy as np
from lpipsPyTorch import lpips

from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights

from utils.system_utils import searchForMaxIteration

def direct_collate(x):
    return x

@torch.no_grad()
def render_set(args, scene, pipe, out_dir, eval):
    render_path = os.path.join(out_dir,"render")
    gt_path = os.path.join(out_dir,"gt")
    os.makedirs(render_path,exist_ok=True)
    os.makedirs(gt_path,exist_ok=True)

    psnr_test = 0.0
    ssims = 0.0
    lpipss = 0.0
    
    aerial_ssims = []
    aerial_psnrs = []
    aerial_lpipss = []
    
    street_ssims = []
    street_psnrs = []
    street_lpipss = []

    cameras = scene.getTestCameras() if eval else scene.getTrainCameras()
    
    cameras = sorted(cameras, key = lambda x : x.colmap_id)

    for idx, viewpoint in tqdm(enumerate(cameras)):
        viewpoint=viewpoint
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()
        
        indices = None

        render_pkg = render(
            viewpoint, 
            scene.gaussians, 
            pipe, 
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
            indices=indices,
            use_trained_exp=args.train_test_exp
            )
        
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)

        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

        alpha_mask = viewpoint.alpha_mask.cuda()

        if args.train_test_exp:
            image = image[..., image.shape[-1] // 2:]
            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
            alpha_mask = alpha_mask[..., alpha_mask.shape[-1] // 2:]

        image_name ='{0:05d}'.format(idx)
        
        torchvision.utils.save_image(gt_image, os.path.join(gt_path, image_name + ".png"))
        try:
            torchvision.utils.save_image(image, os.path.join(render_path, image_name + ".png"))
        except:
            os.makedirs(os.path.dirname(os.path.join(render_path, image_name + ".png")), exist_ok=True)
            torchvision.utils.save_image(image, os.path.join(render_path, image_name + ".png"))
            
        # try:
        #     torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))
        # except:
        #     os.makedirs(os.path.dirname(os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png")), exist_ok=True)
        #     torchvision.utils.save_image(image, os.path.join(render_path, viewpoint.image_name.split(".")[0] + ".png"))

        image *= alpha_mask
        gt_image *= alpha_mask
        
        _psnr = psnr(image, gt_image).mean().double()
        _ssim = ssim(image, gt_image).mean().double()
        _lpips = lpips(image, gt_image, net_type='vgg').mean().double()
        
        psnr_test += _psnr
        ssims += _ssim
        lpipss += _lpips
        
        if "street" in viewpoint.image_path:
            street_psnrs.append(_psnr)
            street_ssims.append(_ssim)
            street_lpipss.append(_lpips)
        else:
            aerial_psnrs.append(_psnr)
            aerial_ssims.append(_ssim)
            aerial_lpipss.append(_lpips)

        torch.cuda.empty_cache()
        
    if eval:
        nums = len(scene.getTestCameras())
    else:
        nums = len(scene.getTrainCameras())
        
    psnr_test /= nums
    ssims /= nums
    lpipss /= nums
    
    aerial_psnr = torch.tensor(aerial_psnrs).mean()
    aerial_ssim = torch.tensor(aerial_ssims).mean()
    aerial_lpips = torch.tensor(aerial_lpipss).mean()
    
    street_psnr = torch.tensor(street_psnrs).mean()
    street_ssim = torch.tensor(street_ssims).mean()
    street_lpips = torch.tensor(street_lpipss).mean()
    
    
    print(f"PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f}")
    print(f"aerial_PSNR: {aerial_psnr:.5f} aerial_SSIM: {aerial_ssim:.5f} aerial_LPIPS: {aerial_lpips:.5f}")
    print(f"street_PSNR: {street_psnr:.5f} street_SSIM: {street_ssim:.5f} street_LPIPS: {street_lpips:.5f}")
    
    with open(os.path.join(out_dir,"metric.txt"), "w") as file:
        file.write(f"PSNR: {psnr_test:.5f} SSIM: {ssims:.5f} LPIPS: {lpipss:.5f} ")
        file.write(f"aerial_PSNR: {aerial_psnr:.5f} aerial_SSIM: {aerial_ssim:.5f} aerial_LPIPS: {aerial_lpips:.5f}")
        file.write(f"street_PSNR: {street_psnr:.5f} street_SSIM: {street_ssim:.5f} street_LPIPS: {street_lpips:.5f}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--out_dir', type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    
    print("Rendering " + args.model_path)

    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    
    try:
        scene = Scene(dataset, gaussians, resolution_scales = [1], load_iteration=-1)
        print("load model using load_iter")
    except:
        load_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
        dataset.pretrained = os.path.join(args.model_path, "point_cloud", "iteration_" + str(load_iter))
        scene = Scene(dataset, gaussians, resolution_scales = [1])
        print("load model using pretrained_model")

    render_set(args, scene, pipe, os.path.join(args.model_path), args.eval)
