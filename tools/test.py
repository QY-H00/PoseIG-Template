# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import pprint
from xmlrpc.client import Boolean

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import compute_epe, compute_poseig, find_J2J, save_dataloader_info, test_on_RI, validate, visualize, compute_RI
from utils.utils import create_logger
import poseig

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    parser.add_argument('--evaluate',
                        default=False,
                        action='store_true',
                        help="Evaluate the model under MSCOCO standard")
    parser.add_argument('--poseig',
                        default=False,
                        action='store_true',
                        help="Compute poseig of the model")
    parser.add_argument('--j2j',
                        default=False,
                        action='store_true',
                        help="Analyze J2J problem")
    parser.add_argument('--epe',
                        default=False,
                        action="store_true",
                        help="Compute EPE of the model")
    parser.add_argument('--vis',
                        default=False,
                        action="store_true",
                        help="visualize a certain sample and joint indicated by --vis_sample and --vis_joint")
    parser.add_argument('--load_ig',
                        default=False,
                        action="store_true",
                        help="Directly loads ig if it's already computed.")
    parser.add_argument('--save_ig',
                        default=False,
                        action="store_true",
                        help="Save ig after it's computed.")
    parser.add_argument('--vis_sample',
                        type=int,
                        default=-1)
    parser.add_argument('--vis_joint',
                        type=int,
                        default=-1)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    
    # save_dataloader_info(valid_loader, model, final_output_dir)
    

    if args.evaluate:
        validate(cfg, valid_loader, valid_dataset, model, criterion,
                final_output_dir, tb_log_dir)
    
    if args.poseig:
        compute_poseig(valid_loader, model, final_output_dir, load_ig=args.load_ig, save_ig=args.save_ig)
    
    if args.j2j:
        find_J2J(valid_loader, model, final_output_dir)
        test_on_RI(cfg, valid_loader, valid_dataset, model, criterion,
          final_output_dir, tb_log_dir)
        compute_RI(valid_loader, model, final_output_dir)
    
    if args.epe:
        compute_epe(valid_loader, model, final_output_dir)
    
    if args.vis:
        if args.vis_sample == -1 or args.vis_joint == -1:
            raise Exception("Please indicate the sample and the joint that you want to visualize.")
        visualize(valid_loader, model, final_output_dir, sample_idx=args.vis_sample, joint=args.vis_joint)
    
    
    



if __name__ == '__main__':
    main()
