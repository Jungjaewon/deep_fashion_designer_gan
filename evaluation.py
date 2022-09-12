import argparse
import os
import os.path as osp

from eval_part.eval_inception_score import Polyvore
from eval_part.eval_inception_score import inception_score

from eval_part.eval_fid import calculate_fid_given_paths

from eval_part.eval_lpips import calculate_lpips

from eval_part.eval_psnr_ssim_mse import calculate_psnr_ssim_mse

# FID https://github.com/mseitzer/pytorch-fid
# Inception score https://github.com/sbarratt/inception-score-pytorch


def eval_func(fp=None, gpu='0'):
    for img_size in [256]:  # , 299
        gt_mean, gt_std, gt_cnt = inception_score(Polyvore(params.orgin_img_dir, params.i_score_net, img_size=img_size),
                                                  img_size=img_size, gpu=gpu)
        print(f"Calculating Inception Score on GT : {(gt_mean, gt_std, gt_cnt)}, img_size : {img_size}")
        if fp:
            print(f"Calculating Inception Score on GT : {(gt_mean, gt_std, gt_cnt)}, img_size : {img_size}", file=fp)

        f_mean, f_std, f_cnt = inception_score(Polyvore(params.score_dir, params.i_score_net, img_size=img_size),
                                               img_size=img_size, gpu=gpu)
        print(f"Calculating Inception Score on Result : {(f_mean, f_std, f_cnt)}, img_size : {img_size}")
        if fp:
            print(f"Calculating Inception Score on Result : {(f_mean, f_std, f_cnt)}, img_size : {img_size}", file=fp)

    for mode in ['base', 'fashion']:
        fid, cnt_fid = calculate_fid_given_paths(fake_dir=params.score_dir, real_dir=params.orgin_img_dir,
                                                 mode=mode, device=f'cuda:{gpu}')
        print(f'Calculating FID : {fid}, mode : {mode}, cnt_fid : {cnt_fid}')
        if fp:
            print(f'Calculating FID : {fid}, mode : {mode}, cnt_fid : {cnt_fid}', file=fp)

    for net in ['alex']:  # 'vgg', 'squeeze'
        for spatial in [False]:  # True,
            LPIPS, cnt = calculate_lpips(fake_dir=params.score_dir, real_dir=params.orgin_img_dir, net=net,
                                         spatial=spatial, device=f'cuda:{gpu}')
            print(f'Calculating LPIPS {net}, spatial: {spatial} : {LPIPS}, cnt : {cnt}')
            if fp:
                print(f'Calculating LPIPS {net}, spatial: {spatial} : {LPIPS}, cnt : {cnt}', file=fp)

    psnr, ssim, mse, cnt = calculate_psnr_ssim_mse(fake_dir=params.score_dir, real_dir=params.orgin_img_dir)
    print(f'Calculating psnr : {psnr}, ssim : {ssim}, mse : {mse}, cnt : {cnt}')
    if fp:
        print(f'Calculating psnr : {psnr}, ssim : {ssim}, mse : {mse}, cnt : {cnt}', file=fp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_dir', type=str, default='refine_g2_glr3e5_dsv3_t1/results/test_100_refine', help='a directory for evaluation')
    parser.add_argument('--orgin_img_dir', type=str, default='test_img/test', help='a directory for GT')
    parser.add_argument('--i_score_net', type=str, default='inception_v3', choices=['vgg19_bn', 'inception_v3'])
    parser.add_argument('--gpu', type=str, default='0')
    params = parser.parse_args()

    print(f'score_dir : {params.score_dir}')

    gpu = params.gpu

    if '/home/ubuntu/research_j/' in params.score_dir:
        txt_name = params.score_dir.replace('./', '').split(os.sep)[5]
        txt_name = f'exp_report/{txt_name}_report.txt'
        if not osp.exists(txt_name):
            with open(txt_name, 'w') as fp:
                eval_func(fp=fp, gpu=gpu)
    else:
        eval_func(fp=None, gpu=gpu)




