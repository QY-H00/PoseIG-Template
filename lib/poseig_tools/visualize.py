import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl


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

