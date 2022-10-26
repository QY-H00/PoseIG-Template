import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from scipy import stats


def draw_kps(img_np, kps):
    for i, uv in enumerate(kps):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_np, str(i), (uv[0], uv[1]), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(img_np, (uv[0], uv[1]), 2, (0, 0, 255), 3)
    
    return img_np


def visualize_ig(attr, path="attr.jpg", bound_scale=1, fusion=False):
    attr -= attr.min()
    attr /= (attr.max() + 1e-20)
    fig, ax = plt.subplots()
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])

    cmap=plt.get_cmap('RdBu_r')
    cmap_bound = np.abs(attr).max()*bound_scale
    norm = mpl.colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    plt.tight_layout()

    ax.imshow(attr, interpolation='none', norm = norm, cmap=cmap)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    

def blend_input(map, input):
    return Image.blend(map, input, 0.4)


def cv2_to_pil(img):
    image = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    return image


def pil_to_cv2(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return image


def vis_saliency_kde(map, zoomin=4):
    grad_flat = map.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    Y, X = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Z = np.reshape(kernel(positions).T, map.shape)
    Z = Z / Z.max()
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    Img = Image.fromarray(map_color)
    s1, s2 = Img.size
    return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)

