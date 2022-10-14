import torch
import poseig_tools.data_util as data_util


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
        assert target.all() <= 1 and target.all() >= 0

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
    pred_kp = data_util.regress25d(pred_hm)
    gt_kp = data_util.regress25d(gt_hm)
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
    target = torch.exp(-0.3*torch.linalg.norm(pred_kp - gt_kp, axis=-1))
    return target


def create_noise(img):
    '''
    
    Creates the noise for integrating the gradient along the noisy path
    
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

