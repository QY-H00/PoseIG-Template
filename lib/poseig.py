from glob import iglob
import os
import pickle
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpyencoder import NumpyEncoder
from kornia.geometry.dsnt import spatial_softmax_2d, spatial_softargmax_2d


class IG_DB:
    def __init__(self, db_path):
        self.db_path = db_path
        
        self.store_ig_cursor = 0
        self.load_ig_cursor = 0
        self.idx_cursor = 0
        self.ig_path = os.path.join(db_path, "ig")
        self.idx_json = os.path.join(db_path, "idx.json")
        self.idx_dict = {}
        
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

   
    def store_batch_ig(self, batch_ig, on_gpu=True):
        batch_size = batch_ig.shape[0]
        for i in range(batch_size):
            file_name = os.path.join(self.ig_path, f"{self.store_ig_cursor}_ig.pickle")
            ig = batch_ig[i]
            if on_gpu:
                ig = ig.detach().cpu().numpy()
            else:
                ig = ig.numpy()
            with open(file_name, 'wb') as f:
                pickle.dump(ig, f)
            self.store_ig_cursor += 1
    
    
    def load_batch_ig(self, batch_size, on_gpu=True):
        device = torch.device("cuda" if on_gpu else "cpu")
        batch_ig = None
        for i in range(batch_size):
            file_name = os.path.join(self.ig_path, f"{self.load_ig_cursor}_ig.pickle")
            with open(file_name, 'rb') as f:
                ig = pickle.load(f)
            ig = torch.from_numpy(ig).to(device)
            if batch_ig is None:
                batch_ig = torch.zeros(batch_size, ig.shape[0], ig.shape[1], ig.shape[2]).to(device)
            batch_ig[i] = ig
            self.load_ig_cursor += 1
        return batch_ig


    def load_ig(self, idx, on_gpu=True):
        device = torch.device("cuda" if on_gpu else "cpu")
        file_name = os.path.join(self.ig_path, f"{idx}_ig.pickle")
        with open(file_name, 'rb') as f:
            ig = pickle.load(f)
        ig = torch.from_numpy(ig).to(device)
        return ig
            
    
    def record_batch_idx(self, batch_di, batch_fi, batch_li):
        assert batch_di.shape == batch_fi.shape and batch_fi.shape == batch_li.shape and len(batch_li.shape) == 2
        batch_size = batch_di.shape[0]
        for i in range(batch_size):
            di = np.round(batch_di[i], 3)
            fi = np.round(batch_fi[i], 3)
            li = np.round(batch_li[i], 3)
            idx_dict = {"DI": di, "FI": fi, "LI": li}
            self.idx_dict[self.idx_cursor] = idx_dict
            self.idx_cursor += 1
    
    
    def store_idx_json(self):
        with open(self.idx_json, 'w') as f:
            json.dump(self.idx_dict, f, cls=NumpyEncoder)
            
    
    def load_idx_json(self):
        with open(self.idx_json, 'r') as f:
            idx_dict = json.load(f)
        return idx_dict


def create_noise(img):
    '''
    
    Create the noise for integrating the gradient along the noisy path
    
    '''
    stdev = 0.01 * (torch.amax(img, axis=(1, 2, 3)) - torch.amin(img, axis=(1,2,3)))
    bs, c, h, w = img.shape
    noise = torch.zeros((bs, c, h, w))
    for _ in range(10):
        for i in range(len(stdev)):
            if stdev[i] < 0.0001:
                stdev[i] = 0.01
            noise[i] = torch.normal(0, stdev[i], img.shape[1:])
    return noise


def compute_poseig(img, model, back_func, back_info, path, cuda=True, noisy=False, p=1):
    '''
    
    Computes the integrated gradient of a model with respect to a image under provided back propagated function and path.
    
    Args:
        img: an image tensor (B, C, H, W)
        model: a deep-learning model accepts img as input
        back_func: a function, given model & img & back_info, computes the target (B, J), and the model output (B, J) to backward propagate for computing gradient
        back_info: a dict, where the elements are utilzed by back_func
        path: a function, given an image, computes interpolated images (folds, B, C, H, W) and lambda interpolated derivative (folds, B, C, H, W)
        cuda: true if using cuda with provided model
        noisy: true if introducing noise when integrating the gradient
        p: power of original integrated gradient, larger will enlarge the difference of attribution among pixels
        
    Returns:
        ig: a tensor (B, J, H, W), the integrated gradient of model w.r.t. img
    '''
    device = torch.device("cuda" if cuda else 'cpu') 
    model = model.to(device)
    
    interpolated_imgs, interpolated_derivate_lambda = path(img)
    fold, bs, c, h, w = interpolated_imgs.shape
    test_target = back_func(model, img, back_info)
    bs, js = test_target.shape
    ig = torch.zeros([bs, js, c, h, w]).to(device)
    
    for i in range(fold):
        iimg = interpolated_imgs[i].to(device) # iimg refers to interpolated_image
        if noisy:
            noise = create_noise(img)
            iimg = iimg + noise.to(device)
        iimg.requires_grad_(True)
        target = back_func(model, iimg, back_info)

        for j in range(js):
            target[:, j].backward(torch.ones_like(target[:, j]), retain_graph=True)
            igrad = iimg.grad # (B, 3, H, W)
            if torch.any(torch.isnan(igrad)):
                igrad[torch.isnan(igrad)] = 0.0
            # integrated along the path
            ig[:, j] += igrad.reshape(bs, c, h, w) * interpolated_derivate_lambda[i]
            iimg.grad.data.zero_()
        target.backward(torch.ones_like(target))
    
    # average on RGB channel
    ig = torch.sum(ig ** p, axis=-3) # (B, J, 3, H, W) -> (B, J, H, W)
    # get the absolute value of integrated graident for robustness analyze
    ig_abs = torch.abs(ig)
    # normalize the integrated gradient
    ig_max = torch.amax(ig_abs, axis=(-1, -2), keepdim=True)
    ig_norm = ig_abs / ig_max
    return ig_norm


def compute_ig(img, model, back_func, back_info, path, cuda=True, noisy=False, p=1):
    '''
    
    Computes the integrated gradient of a model with respect to a image under provided back propagated function and path.
    
    Args:
        img: an image tensor (B, C, H, W)
        model: a deep-learning model accepts img as input
        back_func: a function, given model & img & back_info, computes the target (B,) to backward propagate for computing gradient
        back_info: a dict, where the elements are utilzed by back_func
        path: a function, given an image, computes interpolated images (folds, B, C, H, W) and lambda interpolated derivative (folds, B, C, H, W)
        cuda: true if using cuda with provided model
        noisy: true if introducing noise when integrating the gradient
        p: power of original integrated gradient, larger will enlarge the difference of attribution among pixels
        
    Returns:
        ig: a tensor (B, H, W), the integrated gradient of model w.r.t. img
    '''
    device = torch.device("cuda" if cuda else 'cpu') 
    model = model.to(device)
    
    interpolated_imgs, interpolated_derivate_lambda = path(img)
    fold, bs, c, h, w = interpolated_imgs.shape
    ig = torch.zeros([bs, c, h, w]).to(device)
    
    for i in range(fold):
        iimg = interpolated_imgs[i].to(device) # iimg refers to interpolated_image
        if noisy:
            noise = create_noise(img)
            iimg = iimg + noise.to(device)
        iimg.requires_grad_(True)
        target = back_func(model, iimg, back_info)
        print(target.shape)
        target.backward(torch.ones_like(target))
        igrad = iimg.grad # (B, 3, H, W)
        if torch.any(torch.isnan(igrad)):
            igrad[torch.isnan(igrad)] = 0.0
        # integrated along the path
        ig += igrad.reshape(bs, c, h, w) * interpolated_derivate_lambda[i]
    
    # average on RGB channel
    ig = torch.sum(ig ** p, axis=1) # (B, 3, H, W) -> (B, H, W)
    # get the absolute value of integrated graident for robustness analyze
    ig_abs = torch.abs(ig)
    # normalize the integrated gradient
    ig_max = torch.amax(ig_abs, axis=(1, 2))
    ig_norm = ig_abs / ig_max[:, None, None]
    return ig_norm


def detection_back_func(model, img, detection_back_info):
    '''
    This is the back propogated function of simple regression model
    
    Args:
    model: a regression model simply accepts the image and outputs the position of each keypoints
    img: an image tensor (B, C, H, W)
    regression_back_info: {"gt_hm": ___}
        where regression_back_info["gt_hm"] is the ground truth location of keypoints
        
    Returns:
    target: a target tensor (B, J) used for back propogating to compute gradient
    '''
    
    pred_hm = model(img)
    gt_hm = detection_back_info['gt_hm']
    pred_kp = regress25d(pred_hm)
    gt_kp = regress25d(gt_hm)
    # epe = compute_EPE(pred_kp, gt_kp)
    target = torch.exp(-0.3*torch.linalg.norm(pred_kp - gt_kp, axis=-1))
    return target


def regression_back_func(model, img, regression_back_info):
    '''
    This is the back propogated function of simple regression model
    
    Args:
    model: a regression model simply accepts the image and outputs the position of each keypoints
    img: an image tensor (B, C, H, W)
    regression_back_info: {"gt_kp": ___}
        where regression_back_info["gt_kp"] is the ground truth location of keypoints
        
    Returns:
    target: a target tensor (B, J) used for back propogating to compute gradient
    '''
    
    pred_kp = model(img)
    gt_kp = regression_back_info['gt_kp']
    # epe = compute_EPE(pred_kp, gt_kp)
    target = torch.exp(-0.3*torch.linalg.norm(pred_kp - gt_kp, axis=-1))
    return target


def isotropic_gaussian_kernel(l, sigma, epsilon=1e-5):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * (sigma + epsilon) ** 2))
    return kernel / np.sum(kernel)


def get_gaussian_kernel(kernel_size=25, sigma=9, channels=3):
    gaussian_kernel = torch.from_numpy(isotropic_gaussian_kernel(kernel_size, sigma)).float()
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def torch_filter(kernel_weight, cuda=True):
    device = torch.torch.device("cuda" if cuda else 'cpu') 
    channels=kernel_weight.shape[0]
    kernel_size = kernel_weight.shape[3]
    padding = (kernel_size - 1) // 2
    filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=padding, padding_mode="reflect")
    filter.weight.data = kernel_weight.to(device)
    filter.weight.requires_grad = False
    return filter


def regress25d(heatmaps, cuda=True):
    joints = heatmaps.shape[1]
    bs = heatmaps.shape[0]
    device = torch.torch.device("cuda" if cuda else 'cpu') 
    beta = torch.mul(torch.ones(joints), 10).view(1, joints, 1).repeat(bs, 1, 1).to(device)
    bs = heatmaps.shape[0]
    uv_heatmaps = spatial_softmax_2d(heatmaps, beta)
    coord_out = spatial_softargmax_2d(uv_heatmaps, normalized_coordinates=False)
    coord_out_final = coord_out.clone()
    
    return coord_out_final.view(bs, joints, 2)


def batch_gaussian_path(batch_tensor_image, l=31, fold=50, sigma=19, cuda=True):
    device = torch.device("cuda" if cuda else 'cpu') 
    bs, c, h, w = batch_tensor_image.shape
    kernel_interpolation = torch.zeros((fold + 1, 3, 1, l, l)).to(device)
    image_interpolation = torch.zeros((fold, bs, c, h, w)).to(device)
    lambda_derivative_interpolation = torch.zeros((fold, bs, c, h, w)).to(device)
    sigma_interpolation = np.linspace(sigma, 0, fold + 1)
    for i in range(fold + 1):
        kernel_interpolation[i] = get_gaussian_kernel(sigma=sigma_interpolation[i], kernel_size=l)
    for i in range(fold):
        image_interpolation[i] = torch_filter(kernel_interpolation[i]).forward(batch_tensor_image)
        lambda_derivative_interpolation[i] = torch_filter((kernel_interpolation[i + 1] - kernel_interpolation[i]) * fold).forward(batch_tensor_image)
    return image_interpolation, lambda_derivative_interpolation


def compute_EPE(pr_kps, gt_kps):
    return torch.linalg.norm(pr_kps - gt_kps, axis=-1)


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))


def compute_DI(ig):
    '''
    Computes the diffusion index of given integrated graident map.
    
    Args:
    ig: the integrated gradient tensor in size (B, J, H, W)
    
    Returns:
    di: diffusion index tensor in size (B, J)
    '''
    bs, js = ig.shape[0], ig.shape[1]
    grad = ig.reshape(bs, js, -1)
    diffusions = np.ones((bs, js))
    for b in range(bs):
        for j in range(js):
            gini_index = gini(grad[b, j].cpu().detach().numpy())
            diffusion_index = (1 - gini_index) * 100
            diffusions[b, j] = diffusion_index
    di = np.nan_to_num(diffusions)
    return di


def torch_dilation(mask, size=5):
    '''
    Dilates a mask with pixels of provided size.
    
    Args:
    mask: a mask tensor in size (B, 1, H, W)
    size: an integer, dilation size. For example, if size=2, it means expanding the mask with 2 pixels outer the orginal mask
    
    Returns:
    d_mask: a dilated mask in size (B, 1, H, W)
    '''
    layer = nn.MaxPool2d(size, padding = (size - 1) // 2, stride=1)
    d_mask = layer(mask)
    return d_mask


def compute_FI(ig, mask, cuda=True):
    '''
    Computes the foreground index with provided mask and provided heatmap.
    
    Args:
    ig: an integrated gradient tensor in size (B, J, H, W)
    mask: a mask tensor in size (B, 1, H, W)
    
    Returns:
    fi: foreground index tensor in size (B, J)
    '''
    device = torch.device("cuda" if cuda else "cpu")
    
    # Normalize the ig to make the sum of each ig map is 1
    grad_sum = torch.sum(ig, axis=(-1, -2), keepdim=True)
    ig = ig / grad_sum
    mask = torch_dilation(mask)
    
    pixels_count = (mask.shape[-1]) * (mask.shape[-2])
    mask_sum = torch.sum(mask, axis=(-1, -2)) # (B, 1)
    fi = torch.sum(ig * mask, axis=(-1, -2)) # (B, J)
    fi = fi * pixels_count / mask_sum
    fi = torch.nan_to_num(fi)
    
    return fi.detach().cpu().numpy()


def compute_LI(ig, hm, cuda=True):
    '''
    Computes the foreground index with provided mask and provided heatmap.
    
    Args:
    ig: an integrated gradient tensor in size (B, J, H, W)
    hm: a heatmap tensor in size (B, J, H, W)
    
    Returns:
    li: localization index tensor in size (B, J)
    '''
    
    device = torch.device("cuda" if cuda else "cpu")
    
    # Normalize the ig to make the sum of each ig map is 1
    grad_sum = torch.sum(ig, axis=(-1, -2), keepdim=True)
    ig = ig / grad_sum
    
    pixels_count = (hm.shape[-1]) * (hm.shape[-2])
    hm_sum = torch.sum(hm, axis=(-1, -2))
    li = torch.sum(ig * hm, axis=(-1, -2)) 
    li = li * pixels_count / hm_sum
    li = torch.nan_to_num(li)
    
    return li.detach().cpu().numpy()


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


def compute_indices_with_largest_epes(target_dir, t=10):
    file_path = os.path.join(target_dir, 'epes.pickle')
    with open(file_path, 'rb') as f:
        epes, targets = pickle.load(f)
    
    targets = targets.reshape(-1, )
    epes = epes.reshape(-1,)
    masked_epes = epes * targets
    k = len(epes) - t
    # selected_indices = np.argpartition(epes, k)[k-5:]
    selected_indices = np.argpartition(masked_epes, k)[k-5:]
    # print(selected_indices)
    # print(53015 / 17, 53015 % 17)
    # print(targets[masked_selected_indices])
    # assert False
    sample_indices = (selected_indices / 17).astype(np.int64)
    joint_indices = selected_indices % 17
    return sample_indices, joint_indices


def canonize_ig(ig):
    '''
    Canonize the integrated gradient. Basically, we first transform each entry in ig to its order over all entries.
    For example:
    [[0.4, 0.1], [0.2, 0.3]] will be [[3, 0], [1, 2]]. Since 0.1 is the 1st smallest entry, 0.2 is the 2nd smallest entry and so on.
    Then we normalize the maximum entry into 1. In the example above: it becomes [[1, 0], [1/3, 2/3]].
    
    Args:
    ig: the integrated gradient map in (B, J, H, W)
    
    Returns:
    canonical_ig: each entry encodes the order of same entry in orginal ig overall all entries.
    '''
    bs, js, h, w = ig.shape
    ig = ig.reshape(bs, js, -1)
    sorted, indices = ig.sort(dim=-1)
    canonical_ig = indices.argsort(dim=-1).float()
    canonical_ig /= (torch.amax(canonical_ig, dim=-1, keepdim=True) + 1e-5)
    canonical_ig = canonical_ig.reshape(bs, js, h, w)
    return canonical_ig


def compute_dependency_index(jt_1, jt_2, img, model, back_func, back_info, path, masked_image_output=False):
    '''
    This function computes the correlational index between joint A jt_1 and joint B jt_2 of a certain example.
    
    Args:
    jt_1: index of the first joint
    jt_2: index of the second joint
    img: the input image in batch (1, 3, h, w)
    model: the pose estimation model
    back_func: same as compute_ig
    back_info: same as compute_ig
    path: same as compute_ig
    
    Returns:
    dependency index of jt_1 w.r.t. jt_2, specifically, how jt_1's precision depends on the important pixels when computing jt_2's precision
    '''
    output = {}
    assert img.shape[0] == 1
    pr_kp = regress25d(model(img))
    gt_kp = regress25d(back_info['gt_hm'])
    igs = compute_poseig(img, model, back_func, back_info, path) # (1, j, h, w)
    # igs = torch.rand_like(igs)
    igs = canonize_ig(igs)
    # igs = torch_dilation(igs, size=3)
    js = igs.shape[1]
    assert jt_1 < js and jt_2 < js
    
    # Remove batch dimension
    targets = pr_kp[0]
    target_1 = targets[jt_1]
    target_2 = targets[jt_2]
    igs = igs[0]
    ig_1 = igs[jt_1]
    ig_2 = igs[jt_2] # (h, w)
    
    img_mean = torch.mean(img, (0,1,2,3), keepdim=True)
    # mask img by setting the weight of ig_2 where its maximum is 1
    ig_2_normed = ig_2 / (torch.amax(ig_2) + 1e-5)
    masked_2_img = img * (1 - ig_2_normed) + img_mean * ig_2_normed
    
    # compute dependency index of jt_1 with respect to jt_2
    targets_1_masked = regress25d(model(masked_2_img))[0]
    target_1_masked = targets_1_masked[jt_1]
    dependency_index_1_to_2 = target_1_masked / (target_1_masked + target_1 + 1e-5) # larger means jt_1 depends more on jt_2
    
    # mask img by setting the weight of ig_2 where its maximum is 1
    ig_1_normed = ig_1 / (torch.amax(ig_1) + 1e-5)
    masked_1_img = img - ig_1_normed * img + img_mean * ig_1_normed
    
    # compute dependency index of jt_1 with respect to jt_2
    targets_2_masked = regress25d(model(masked_1_img))[0]
    target_2_masked = targets_2_masked[jt_2]
    dependency_index_2_to_1 = target_2_masked / (target_2_masked + target_2 + 1e-5) # larger means jt_2 depends more on jt_1
    # return dependency_index_1_to_2, dependency_index_2_to_1
    output = {
        "gt_1": gt_kp[0][jt_1] * 4, # Reshrink to original 256 x 192 size
        "gt_2": gt_kp[0][jt_2] * 4,
        "pr_1": target_1 * 4,
        "pr_1m2": target_1_masked * 4,
        "pr_2": target_2 * 4,
        "pr_2m1": target_2_masked * 4,
        "ig_1": ig_1,
        "ig_2": ig_2,
        "img_m1": masked_1_img,
        "img_m2": masked_2_img
    }
    output["epe_1"] = torch.linalg.norm(output["pr_1"] - output["gt_1"])
    output["epe_1m2"] = torch.linalg.norm(output["pr_1m2"] - output["gt_1"])
    output["epe_2"] = torch.linalg.norm(output["pr_2"] - output["gt_2"])
    output["epe_2m1"] = torch.linalg.norm(output["pr_2m1"] - output["gt_2"])
    return output
        
        
def compute_RI(ig, hm, jt_1, jt_2):
    '''
    This index is different from other index, ig and hm here is not in batch.
    '''
    ig_1 = ig[jt_1].unsqueeze(0).unsqueeze(0)
    hm_2 = hm[jt_2].unsqueeze(0).unsqueeze(0)
    rl_1to2 = compute_LI(ig_1, hm_2)[0, 0]
    return rl_1to2
    
    
