import torch
import torch.nn as nn
import numpy as np


def compute_EPE(pr_kps, gt_kps):
    '''
    Computes the euclidean distance between each pair in two sets of keypoints.
    
    Args:
    pr_kps, torch tensor with size (B, J, 2 or 3) / (J, 2 or 3)
    gt_kps, torch tensor with size (B, J, 2 or 3) / (J, 2 or 3)
    
    Returns:
    torch tensor with size (B, J) or (J)
    '''
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


def compute_RI(ig, hm, jt_1, jt_2):
    '''
    This index is different from other index, ig and hm here is not in batch.
    '''
    ig_1 = ig[jt_1].unsqueeze(0).unsqueeze(0)
    hm_2 = hm[jt_2].unsqueeze(0).unsqueeze(0)
    rl_1to2 = compute_LI(ig_1, hm_2)[0, 0]
    return rl_1to2

    
