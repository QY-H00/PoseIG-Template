import torch
import torch.nn as nn
import numpy as np

def isotropic_gaussian_kernel(l, sigma, epsilon=1e-5):
    '''
    Creates a gaussian kernel (numpy array) in size [l, l].
    '''
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * (sigma + epsilon) ** 2))
    return kernel / np.sum(kernel)


def get_gaussian_kernel(kernel_size=25, sigma=9, channels=3):
    '''
    Creates a gaussian kernel (torch tensor) in size [c, 1, l, l].
    '''
    gaussian_kernel = torch.from_numpy(isotropic_gaussian_kernel(kernel_size, sigma)).float()
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def torch_filter(kernel_weight, cuda=True):
    '''
    Creates a convolution layer (torch.nn.layer) with kernel_weight.
    '''
    device = torch.torch.device("cuda" if cuda else 'cpu') 
    channels=kernel_weight.shape[0]
    kernel_size = kernel_weight.shape[3]
    padding = (kernel_size - 1) // 2
    filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=padding, padding_mode="reflect")
    filter.weight.data = kernel_weight.to(device)
    filter.weight.requires_grad = False
    return filter


def batch_gaussian_path(batch_tensor_image, l=31, fold=50, sigma=19, cuda=True):
    '''
    Creates the gaussian path using batch_tensor_image as the base image. 
    
    Returns:
    image_interpolation, torch tensor with size (fold, bs, c, h, w)
    image_interpolation[0] is batch_tensor_image filtered by gaussian kernel (l, sigma).
    Along the path the image is filtered by a gaussian kernel with less sigma until 0,
    and image_interpolation[fold - 1] is batch_tensor_image itself. 
    
    lambda_derivative_interpolation, torch tensor with size (fold, bs, c, h, w)
    lambda_derivative_interpolation[i] = image_interpolation[i + 1] - image_interpolation[i]
    '''
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

