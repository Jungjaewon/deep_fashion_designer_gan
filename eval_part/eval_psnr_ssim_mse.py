import os
import os.path as osp

from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import cv2

# https://pythonmana.com/2022/01/202201180503200144.html

def avg(data_list):
    return sum(data_list) / float(len(data_list))


def calculate_psnr_ssim_mse(fake_dir, real_dir):

    mse_list, psnr_list, ssim_list, cnt = list(), list(), list(), 0
    for img_file in os.listdir(fake_dir):

        if osp.exists(osp.join(real_dir, img_file)):

            cnt += 1

            img_0 = cv2.imread(osp.join(fake_dir, img_file), cv2.IMREAD_COLOR)
            img_1 = cv2.imread(osp.join(real_dir, img_file), cv2.IMREAD_COLOR)

            psnr = compare_psnr(image_test=img_0, image_true=img_1)
            ssim = ssim_func(img_0, img_1, channel_axis=2)
            mse = compare_mse(img_0, img_1)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mse_list.append(mse)

    return avg(psnr_list), avg(ssim_list), avg(mse_list), cnt