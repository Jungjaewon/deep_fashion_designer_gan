import os
import os.path as osp

from glob import glob

dfd_base_dir = '/home/ubuntu/research_j/dfd_baseline'
dfd_gan_dir = '/home/ubuntu/research_j/deep_fashion_designer_gan'
dfd_base_list = ['cylcegan_unet_0', 'cylcegan_resent_0', 'pix2pix_resnet_0', 'pix2pix_unet_0']
dfd_base_list = [osp.join(dfd_base_dir, x) for x in dfd_base_list]

#dfd_gan_list = []
#for p in [50, 100, 150, 200]:
#    for e in [0, 1, 2]:
#        dfd_gan_list.append(osp.join(dfd_gan_dir, f'lsgan_gt_percep_{str(p)}_en{str(e)}'))


#dfd_gan_list = ['lsgan_gt_percep_25_en1', 'lsgan_gt_percep_50_en1/', 'lsgan_gt_percep_50_en0/', 'refine_lsgan_gt_percep_50_en1', 'refine_lsgan_gt_percep_50_en1_rlr_1e-5',
#                'refine_lsgan_gt_percep_50_en1_rlr_1e-4_lambdaLR', 'lsgan_gt_only_en1', 'lsgan_gt_percep_50_en1_no_mapping', 'lsgan_gt_percep_50_en2']
dfd_gan_list = ['lsgan_gt_percep_50_en1_vgg_14_27_40']
dfd_gan_list = [osp.join(dfd_gan_dir, x) for x in dfd_gan_list]


if __name__ == '__main__':

    cmd = f'python3 evaluation.py --score_dir '
    for target_dir in dfd_gan_list:
    #for target_dir in dfd_base_list + dfd_gan_list:
        candi_list = glob(osp.join(target_dir, 'results', '*'))
        candi_list = [x for x in candi_list if osp.isdir(x)]
        epoch_list = list()
        for candi in candi_list:
            b_dir_split = osp.basename(candi).split('_')
            for x in b_dir_split:
                if x.isdigit():
                    epoch_list.append(int(x))
        epoch_list = [x for x in epoch_list if isinstance(x, int)]
        epoch = max(epoch_list)

        fake_dir = osp.join(target_dir, 'results', f'test_{epoch}_fake')
        refine_dir = osp.join(target_dir, 'results', f'test_{epoch}_refine')

        if osp.exists(refine_dir):
            t_dir = refine_dir
        else:
            t_dir = fake_dir

        print('*' * 15)
        os.system(cmd + t_dir)
        print('*' * 15)