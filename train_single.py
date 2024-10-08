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

import os
import torch
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import shutil
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    assert os.path.exists(os.path.join(ROOT, '.gitignore'))
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, logger=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                aerial_l1_test = []
                aerial_psnr_test = []
                
                street_l1_test = []
                street_psnr_test = []
                
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    # breakpoint()
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if viewpoint.alpha_mask is not None:
                        alpha_mask = viewpoint.alpha_mask.cuda()
                        image = image*alpha_mask
                        gt_image = gt_image*alpha_mask

                    _l1_loss = l1_loss(image, gt_image).mean().double()
                    _psnr = psnr(image, gt_image).mean().double()
                    l1_test += _l1_loss
                    psnr_test += _psnr
                    
                    if "street" in viewpoint.image_path:
                        street_l1_test.append(_l1_loss)
                        street_psnr_test.append(_psnr)
                    else:
                        aerial_l1_test.append(_l1_loss)
                        aerial_psnr_test.append(_psnr)
                        
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                aerial_l1_test = torch.tensor(aerial_l1_test).mean().double()
                aerial_psnr_test = torch.tensor(aerial_psnr_test).mean().double()
                street_l1_test = torch.tensor(street_l1_test).mean().double()
                street_psnr_test = torch.tensor(street_psnr_test).mean().double()
                      
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                logger.info("\n[ITER {}] Evaluating {}: aerial_L1 {} aerial_PSNR {}".format(iteration, config['name'], aerial_l1_test, aerial_psnr_test))
                logger.info("\n[ITER {}] Evaluating {}: street_L1 {} street_PSNR {}".format(iteration, config['name'], street_l1_test, street_psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        

def direct_collate(x):
    return x

def training(dataset, opt, pipe, saving_iterations, testing_iterations, checkpoint_iterations, checkpoint, debug_from, logger=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    indices = None  
    
    training_generator = DataLoader(scene.getTrainCameras(), num_workers = 8, prefetch_factor = 1, persistent_workers = True, collate_fn=direct_collate)

    iteration = first_iter

    while iteration < opt.iterations + 1:
        for viewpoint_batch in training_generator:
            for viewpoint_cam in viewpoint_batch:
                background = torch.rand((3), dtype=torch.float32, device="cuda")

                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.cuda()
                viewpoint_cam.projection_matrix = viewpoint_cam.projection_matrix.cuda()
                viewpoint_cam.full_proj_transform = viewpoint_cam.full_proj_transform.cuda()
                viewpoint_cam.camera_center = viewpoint_cam.camera_center.cuda()

                if not args.disable_viewer:
                    if network_gui.conn == None:
                        network_gui.try_connect()
                    while network_gui.conn != None:
                        try:
                            net_image_bytes = None
                            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                            if custom_cam != None:
                                if keep_alive:
                                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, indices = indices)["render"]
                                else:
                                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, indices = indices)["depth"].repeat(3, 1, 1)
                                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                            network_gui.send(net_image_bytes, dataset.source_path)
                            if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                                break
                        except Exception as e:
                            network_gui.conn = None

                iter_start.record()

                gaussians.update_learning_rate(iteration)

                # Every 1000 its we increase the levels of SH up to a maximum degree
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()

                # Render
                if (iteration - 1) == debug_from:
                    pipe.debug = True
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, indices = indices, use_trained_exp=True)
                image, invDepth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                # Loss
                gt_image = viewpoint_cam.original_image.cuda()
                if viewpoint_cam.alpha_mask is not None:
                    alpha_mask = viewpoint_cam.alpha_mask.cuda()
                    image *= alpha_mask
                
                Ll1 = l1_loss(image, gt_image)
                Lssim = (1.0 - ssim(image, gt_image))
                photo_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim 
                loss = photo_loss.clone()
                Ll1depth_pure = 0.0
                if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                    mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                    depth_mask = viewpoint_cam.depth_mask.cuda()

                    Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
                    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0

                loss.backward()
                iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    ema_loss_for_log = 0.4 * photo_loss.item() + 0.6 * ema_loss_for_log
                    ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
                    if iteration % 10 == 0:
                        if viewpoint_cam.alpha_mask is not None:
                            psnr_log = psnr(image*alpha_mask, gt_image*alpha_mask).mean().double()
                        else:
                            psnr_log = psnr(image, gt_image).mean().double()
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "Size": f"{gaussians._xyz.size(0)}", "psnr":f"{psnr_log:.{3}f}"})
                        progress_bar.update(10)

                    # Log and save
                    # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), logger)
                    if (iteration in saving_iterations):
                        logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                        scene.save(iteration)
                        print("peak memory: ", torch.cuda.max_memory_allocated(device='cuda'))

                    if iteration == opt.iterations:
                        progress_bar.close()
                        return

                    # Densification
                    if iteration < opt.densify_until_iter:
                        # Keep track of max radii in image-space for pruning
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii)
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent)
                        
                        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                            print("-----------------RESET OPACITY!-------------")
                            gaussians.reset_opacity()

                    # Optimizer step
                    if iteration < opt.iterations:
                        gaussians.exposure_optimizer.step()
                        gaussians.exposure_optimizer.zero_grad(set_to_none = True)

                        if gaussians._xyz.grad != None and gaussians.skybox_locked:
                            gaussians._xyz.grad[:gaussians.skybox_points, :] = 0
                            gaussians._rotation.grad[:gaussians.skybox_points, :] = 0
                            gaussians._features_dc.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._features_rest.grad[:gaussians.skybox_points, :, :] = 0
                            gaussians._opacity.grad[:gaussians.skybox_points, :] = 0
                            gaussians._scaling.grad[:gaussians.skybox_points, :] = 0

                        if gaussians._opacity.grad != None:
                            relevant = (gaussians._opacity.grad.flatten() != 0).nonzero()
                            relevant = relevant.flatten().long()
                            if(relevant.size(0) > 0):
                                gaussians.optimizer.step(relevant)
                            else:
                                gaussians.optimizer.step(relevant)
                                print("No grads!")
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    if not args.skip_scale_big_gauss:
                        with torch.no_grad():
                            vals, _ = gaussians.get_scaling.max(dim=1)
                            violators = vals > scene.cameras_extent * 0.02
                            if gaussians.scaffold_points is not None:
                                violators[:gaussians.scaffold_points] = False
                            gaussians._scaling[violators] = gaussians.scaling_inverse_activation(gaussians.get_scaling[violators] * 0.8)
                        
                    if (iteration in checkpoint_iterations):
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                        torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                    iteration += 1

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # update args
    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)
    
    op.densify_until_iter = op.iterations // 2
    op.position_lr_max_steps = op.iterations
    
    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != op.iterations:
        args.test_iterations.append(op.iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(10000, op.iterations + 1, 10000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != op.iterations:
        args.save_iterations.append(op.iterations)
    
    os.makedirs(lp.model_path, exist_ok=True)
    logger = get_logger(lp.model_path)
    
    logger.info("Optimizing " + args.model_path)

    if args.eval and args.exposure_lr_init > 0 and not args.train_test_exp: 
        print("Reconstructing for evaluation (--eval) with exposure optimization on the train set but not for the test set.")
        print("This will lead to high error when computing metrics. To optimize exposure on the left half of the test images, use --train_test_exp")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # save code
    try:
        saveRuntimeCode(os.path.join(lp.model_path, 'backup'))
    except:
        print(f'save code failed~')
    

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, args.save_iterations, args.test_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger)

    # All done
    logger.info("\nTraining complete.")
