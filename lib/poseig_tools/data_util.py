import os
import pickle
import json
from numpyencoder import NumpyEncoder

import torch
import numpy as np

from kornia.geometry.dsnt import spatial_softmax_2d, spatial_softargmax_2d


class IG_DB:
    def __init__(self, db_path):
        self.db_path = db_path
        
        self.ig_path = os.path.join(db_path, "ig")
        self.idx_json = os.path.join(db_path, "idx.json")
        self.epe_json = os.path.join(db_path, "epe_json")
        self.idx_dict = {}
        self.epe_dict = {}
        
        if not os.path.exists(db_path):
            try:
                os.makedirs(db_path)
            except Exception:
                print('Fail to make {}'.format(db_path))
        
        if not os.path.exists(self.ig_path):
            try:
                os.makedirs(self.ig_path)
            except Exception:
                print('Fail to make {}'.format(self.ig_path))

   
    def store_batch_ig(self, batch_ig, cursor, on_gpu=True):
        batch_size = batch_ig.shape[0]
        for i in range(batch_size):
            file_name = os.path.join(self.ig_path, f"{cursor}_ig.pickle")
            ig = batch_ig[i]
            if on_gpu:
                ig = ig.detach().cpu().numpy()
            else:
                ig = ig.numpy()
            with open(file_name, 'wb') as f:
                pickle.dump(ig, f)
            cursor += 1
        return cursor
    
    
    def load_batch_ig(self, batch_size, cursor, on_gpu=True):
        device = torch.device("cuda" if on_gpu else "cpu")
        batch_ig = None
        for i in range(batch_size):
            file_name = os.path.join(self.ig_path, f"{cursor}_ig.pickle")
            with open(file_name, 'rb') as f:
                ig = pickle.load(f)
            ig = torch.from_numpy(ig).to(device)
            if batch_ig is None:
                batch_ig = torch.zeros(batch_size, ig.shape[0], ig.shape[1], ig.shape[2]).to(device)
            batch_ig[i] = ig
            cursor += 1
        return batch_ig, cursor
    
    
    def load_ig(self, cursor, on_gpu=True):
        device = torch.device("cuda" if on_gpu else "cpu")
        file_name = os.path.join(self.ig_path, f"{cursor}_ig.pickle")
        with open(file_name, 'rb') as f:
            ig = pickle.load(f)
        ig = torch.from_numpy(ig).to(device)
        return ig
            
    
    def record_batch_idx(self, batch_di, batch_fi, batch_li, target_weight, cursor):
        assert batch_di.shape == batch_fi.shape and batch_fi.shape == batch_li.shape and len(batch_li.shape) == 2
        batch_size = batch_di.shape[0]
        for i in range(batch_size):
            di = np.round(batch_di[i], 3)
            fi = np.round(batch_fi[i].astype("float64"), 3)
            li = np.round(batch_li[i].astype("float64"), 3)
            idx_dict = {"DI": di, "FI": fi, "LI": li, "weight": target_weight[i]}
            self.idx_dict[cursor] = idx_dict
            cursor += 1
        return cursor
    
    
    def save_idx_json(self):
        with open(self.idx_json, 'w') as f:
            json.dump(self.idx_dict, f, cls=NumpyEncoder)
            
    
    def load_idx_json(self):
        with open(self.idx_json, 'r') as f:
            idx_dict = json.load(f)
        return idx_dict
 

    def record_batch_epe(self, batch_epe, target_weight, cursor):
        assert len(batch_epe.shape) == 2
        batch_size = batch_epe.shape[0]
        for i in range(batch_size):
            epe = np.round(batch_epe[i].astype("float64"), 3)
            epe_dict = {"EPE": epe, "weight": target_weight[i]}
            self.epe_dict[cursor] = epe_dict
            cursor += 1
        return cursor   


    def save_epe_json(self):
        with open(self.epe_json, 'w') as f:
            json.dump(self.epe_dict, f, cls=NumpyEncoder)
            
    
    def load_epe_json(self):
        with open(self.epe_json, 'r') as f:
            epe_dict = json.load(f)
        return epe_dict


def gen_heatmap(img, pt, sigma):
    """
    Generates heatmap based on pt coord.
    Noted that this API is used in dataloader.

    Args:
    img: original heatmap, zeros, np (H,W) float32
    pt: keypoint coord, np (2,) int32
    sigma: guassian sigma, float
    
    Returns:
    generated heatmap, np (H, W) each pixel values id a probability
    flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1


def regress25d(heatmaps, beta_weight=10, cuda=True):
    '''
    Simulates the argmax function using soft-argmax function.
    '''
    joints = heatmaps.shape[1]
    bs = heatmaps.shape[0]
    device = torch.torch.device("cuda" if cuda else 'cpu') 
    beta = torch.mul(torch.ones(joints), beta_weight).view(1, joints, 1).repeat(bs, 1, 1).to(device)
    bs = heatmaps.shape[0]
    uv_heatmaps = spatial_softmax_2d(heatmaps, beta)
    coord_out = spatial_softargmax_2d(uv_heatmaps, normalized_coordinates=False)
    coord_out_final = coord_out.clone()
    
    return coord_out_final.view(bs, joints, 2)


def check_J2J(pr_kp, gt_kp, target_weight, epsilon=10):
    pairs = []
    count = 0
    swap_count = 0
    for jt1_idx in range(len(pr_kp)):
        for jt2_idx in range(jt1_idx + 1, len(pr_kp)):
            flag = False
            if target_weight[jt1_idx, 0] * target_weight[jt2_idx, 0] == 0:
                continue
            if torch.linalg.norm(pr_kp[jt1_idx] - gt_kp[jt1_idx]) > epsilon and torch.linalg.norm(pr_kp[jt1_idx] - gt_kp[jt2_idx]) < epsilon:
                pairs.append([jt1_idx, jt2_idx])
                count += 1
                flag = True
            if torch.linalg.norm(pr_kp[jt2_idx] - gt_kp[jt2_idx]) > epsilon and torch.linalg.norm(pr_kp[jt2_idx] - gt_kp[jt1_idx]) < epsilon:
                pairs.append([jt2_idx, jt1_idx])
                count += 1
                if flag:
                    swap_count += 1
    return pairs, count, swap_count
    

def torch_to_cv2_img(img_tensor):
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np) + 1e-5)
    img_np = img_np * 255
    img_np = img_np.astype(np.uint8).copy()
    return img_np


def torch_to_cv2_kp(kps, is_cuda=True):
    kps = kps.detach().cpu() if is_cuda else kps
    kps = kps.numpy().astype(np.uint8).copy()
    return kps
    
