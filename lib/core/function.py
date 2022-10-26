# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import time
import logging
import os

import numpy as np
import torch
import torchvision

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
import poseig_tools as poseig
from progress.bar import Bar
import cv2
import json
from numpyencoder import NumpyEncoder


logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred*4, output,
                              prefix)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                            val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()


                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5


            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                                target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred*4, output,
                                    prefix)

        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums
        )

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

        return perf_indicator


def compute_poseig(val_loader, model, output_dir, load_ig=False, save_idx=True, save_ig=True):
    # switch to evaluate mode
    selected_batch_idx = [157, 321, 42, 158, 366, 143, 280, 305, 224, 386, 227, 228, 232, 296, 55, 344, 351, 18, 242, 34]
    batch_size = 16
    model.eval()

    length = len(selected_batch_idx)
    DI = AverageMeter()
    FI = AverageMeter()
    LI = AverageMeter()
    db = poseig.IG_DB(os.path.join(output_dir, "IG_DB"))

    count = 0
    bar = Bar(f'\033[31m Process \033[0m', max=length)
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        if i not in selected_batch_idx:
            continue
        count += 1
        last = time.time()
        if load_ig:
            ig, _ = db.load_batch_ig(batch_size=input.shape[0], cursor=i * batch_size)
        else:
            back_info = {"gt_hm": target.to("cuda")}
            ig = poseig.compute_poseig(input.to("cuda"), model, poseig.detection_back_func, back_info, poseig.batch_gaussian_path)
        
        target_weight = target_weight[:, :, 0].detach().cpu().numpy() # (B, J, 1) -> (B, J)
        valid_count = np.sum(target_weight)
        
        # Compute index
        di_bh = poseig.compute_DI(ig)
        di_bh = di_bh * target_weight # (B, J)
        
        hm = torchvision.transforms.Resize((meta['mask'].shape[-2], meta['mask'].shape[-1])).to("cuda").forward(target).to("cuda")
        li_bh = poseig.compute_LI(ig, hm)
        li_bh = li_bh * target_weight
        
        mask = torch.unsqueeze(meta['mask'], 1).to("cuda")
        fi_bh = poseig.compute_FI(ig, mask)
        fi_bh = fi_bh * target_weight
        
        torch.cuda.empty_cache()
        
        DI.update(di_bh.mean(), valid_count)
        FI.update(fi_bh.mean(), valid_count)
        LI.update(li_bh.mean(), valid_count)
        bar.suffix = (
                    '({batch}/{size}) '
                    'DI: {DI:.4f} | '
                    'FI: {FI:.4f} | '
                    'LI: {LI:.4f} | '
                    'time cost: {cost:.4f} | '
                ).format(
                    batch=i + 1,
                    size=length,
                    DI=DI.avg,
                    FI=FI.avg,
                    LI=LI.avg,
                    cost=time.time() - last
                )
        db.record_batch_idx(di_bh, fi_bh, li_bh, target_weight, cursor=i * batch_size)
        if save_ig:
            db.store_batch_ig(ig, cursor=i * batch_size)
        bar.next()
    if save_idx:
        db.save_idx_json()
    bar.finish()

    
def visualize(val_loader, model, output_dir, sample=3633, joint=5):
    # switch to evaluate mode
    model.eval()
    
    length = len(val_loader)
    db = poseig.IG_DB(os.path.join(output_dir, "IG_DB"))
    
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        batch_size = input.shape[0]

        model = model.to("cuda")
        
        if i == sample // batch_size:
            ig = db.load_ig(sample)
            target_dir = db.db_path
            prefix = os.path.join(target_dir, 'imgs')
            ig_file_name = '{}_{}_ig.jpg'.format(sample, joint)
            ig_file_name = os.path.join(prefix, ig_file_name)
            img_file_name = '{}_img.jpg'.format(sample)
            img_file_name = os.path.join(prefix, img_file_name)
            kde_file_name = '{}_{}_kde.jpg'.format(sample, joint)
            kde_file_name = os.path.join(prefix, kde_file_name)
            poseig.visualize_ig(ig.detach().cpu(), path=ig_file_name)
            
            start = time.time()
            print(f"\nStart kernel density estimation, the process requires nearly 1 minutes...")
            kde = poseig.vis_saliency_kde(ig.detach().cpu(), zoomin=1)
            print(f"\nFinish kde, time cost: {time.time() - start}")
            
            img_np = poseig.torch_to_cv2_img(input[sample % batch_size])
            blend_kde = poseig.blend_input(kde.convert("RGB"), poseig.cv2_to_pil(img_np))
            cv2.imwrite(kde_file_name, poseig.pil_to_cv2(blend_kde))
            
            kps = poseig.regress25d(model(input))[sample % batch_size] * 4
            # gt_kps = poseig.regress25d(target.to("cuda"))[sample % batch_size] * 4
            img_np = poseig.draw_kps(img_np, poseig.torch_to_cv2_kp(kps))
            cv2.imwrite(img_file_name, img_np)
            break


def compute_epe(val_loader, model, output_dir, mode="default", save_epe=True):
    assert mode in ["default", "white", "black", "blur"]
    
    # switch to evaluate mode
    model.eval()

    length = len(val_loader)
    mepe = AverageMeter()
    db = poseig.IG_DB(os.path.join(output_dir, "IG_DB"))
    cursor = 0
    
    bar = Bar(f'\033[31m Process \033[0m', max=length)
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        # compute output
        last = time.time()

        model = model.to("cuda")
        
        if mode == "white":
            input = torch.ones_like(input) * 255
        if mode == "black":
            input = torch.zeros_like(input)
        if mode == "blur":
            kernel = poseig.get_gaussian_kernel(sigma=19, kernel_size=30)
            input = poseig.torch_filter(kernel).forward(input.to("cuda"))
        
        pr_hm = model(input.to("cuda"))
        pr_kp = poseig.regress25d(pr_hm)
        gt_kp = poseig.regress25d(target.to("cuda"))
        epe = poseig.compute_EPE(pr_kp, gt_kp).detach().cpu().numpy()
        target_weight = target_weight[:, :, 0].detach().cpu().numpy()
        epe = epe * target_weight
        
        mepe.update(epe.mean(), np.sum(target_weight))
        bar.suffix = (
                    '({batch}/{size}) '
                    'mepe: {mepe:.4f} | '
                    'time cost: {cost:.4f} | '
                ).format(
                    batch=i + 1,
                    size=length,
                    mepe=mepe.avg,
                    cost=time.time() - last
                )
        bar.next()
        cursor = db.record_batch_epe(epe, target_weight, cursor)
    
    if save_epe:
        db.save_epe_json()  
    bar.finish()


def find_J2J(val_loader, model, output_dir, epsilon=10):
    # switch to evaluate mode
    model.eval()

    length = len(val_loader)
    count = 0
    swap_count = 0
    j2j_json = os.path.join(output_dir, "IG_DB", "j2j.json")
    j2j_dict = {"J2J": {}}

    bar = Bar(f'\033[31m Process \033[0m', max=length)
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        # compute output
        last = time.time()

        model = model.to("cuda")
        
        pr_hm = model(input.to("cuda"))
        pr_kp = poseig.regress25d(pr_hm) * 4
        gt_kp = poseig.regress25d(target.to("cuda")) * 4
        epsilon = 10
        
        for k in range(pr_kp.shape[0]):
            sample = i * pr_kp.shape[0] + k
            pairs, _count, _swap_count = poseig.check_J2J(pr_kp[k], gt_kp[k], target_weight[k], epsilon=epsilon)
            for pair in pairs:
                jt1_idx = pair[0]
                jt2_idx = pair[1]
                j2j_dict["J2J"][sample*17**2 + jt1_idx*17 + jt2_idx] = [sample, jt1_idx, jt2_idx]
            count += _count
            swap_count += _swap_count
                            
        bar.suffix = (
                    '({batch}/{size}) '
                    'count_J2J: {count} | '
                    'count_swap_J2J: {swap_count} | '
                    'time cost: {cost:.4f} | '
                ).format(
                    batch=i + 1,
                    size=length,
                    count = count,
                    swap_count = swap_count,
                    cost=time.time() - last
                )
        bar.next()

    bar.finish()
    with open(j2j_json, 'w') as f:
            json.dump(j2j_dict, f, cls=NumpyEncoder)


def test_on_RI(val_loader, model, output_dir):
    # switch to evaluate mode
    model.eval()

    length = len(val_loader)
    db = poseig.IG_DB(os.path.join(output_dir, "IG_DB"))
    j2j_json = os.path.join(output_dir, "IG_DB", "j2j.json")
    with open(j2j_json, 'r') as f:
        j2j_dict = json.load(f)
    j2j_list = np.array(list(j2j_dict["J2J"].values()))
    j2j_sample = np.random.randint(0, len(j2j_list), size=100)
    non_j2j_sample = np.random.randint(0, len(val_loader)* 17**2, size=100)
    j2j_sample = j2j_list[j2j_sample]
    j2j_idx = j2j_sample[:, 0]
    j2j_jt1 = j2j_sample[:, 1]
    j2j_jt2 = j2j_sample[:, 2]
    non_j2j_idx = non_j2j_sample // 17 ** 2
    non_j2j_jt2 = non_j2j_sample % 17
    non_j2j_jt1 = (non_j2j_sample % 17 ** 2) // 17
    j2j_RI = AverageMeter()
    non_j2j_RI = AverageMeter()

    bar = Bar(f'\033[31m Process \033[0m', max=length)
    for i, (input, target, target_weight, meta) in enumerate(val_loader):
        assert input.shape[0] == 1
        ig = db.load_batch_ig(batch_size=input.shape[0])
        ig = ig[0]
        hm = torchvision.transforms.Resize((meta['mask'].shape[-2], meta['mask'].shape[-1])).to("cuda").forward(target).to("cuda")
        hm = hm[0]
        if i in j2j_idx:
            k = np.where(j2j_idx == i)[0][0]
            ri = poseig.compute_RI(ig, hm, j2j_jt1[k], j2j_jt2[k])
            j2j_RI.update(ri, 1)
        
        if i in non_j2j_idx:
            k = np.where(non_j2j_idx == i)[0][0]
            ri = poseig.compute_RI(ig, hm, non_j2j_jt1[k], non_j2j_jt2[k])
            non_j2j_RI.update(ri, 1)
        
        bar.suffix = (
                    '({batch}/{size}) '
                    'J2J_RI: {J2J_RI:.4f} | '
                    'non_J2J_RI: {non_J2J_RI:.4f} | '
                ).format(
                    batch=i + 1,
                    size=length,
                    J2J_RI=j2j_RI.avg,
                    non_J2J_RI=non_j2j_RI.avg
                )
        bar.next()
    
    bar.finish()


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
