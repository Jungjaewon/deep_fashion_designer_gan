import math
import glob
import os.path as osp
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

"""
def psnr(original, contrast):
    # https://github.com/jackfrued/Python-1/blob/master/analysis/compression_analysis/psnr.py
    mse = np.mean((original - contrast) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR

#https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
"""



def get_result(gt_dir, fake_dir, metric='psnr'):

    assert metric in ['psnr', 'ssim']

    print(f'metric is {metric}')
    print(f'gt_dir is {gt_dir}')
    print(f'fake_dir is {fake_dir}')
    gt_list = sorted(glob.glob(osp.join(gt_dir, '*.jpg')))
    fake_list = sorted(glob.glob(osp.join(fake_dir, '*.jpg')))

    assert len(gt_list) == len(fake_list)

    result = 0

    for i in range(len(gt_list)):

        assert osp.basename(fake_list[i]) == osp.basename(gt_list[i])

        if metric == 'psnr':
            result += psnr(cv2.imread(gt_list[i]), cv2.imread(fake_list[i]))
        else:
            result += ssim(cv2.imread(gt_list[i]), cv2.imread(fake_list[i]))


    return result

