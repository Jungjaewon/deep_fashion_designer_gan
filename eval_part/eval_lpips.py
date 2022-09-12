import os
import os.path as osp
import lpips



def calculate_lpips(fake_dir, real_dir, device='cuda:0', version='0.1', net='alex', spatial=False):
    loss_fn = lpips.LPIPS(net=net, version=version, spatial=spatial)
    lpips_dist, cnt = list(), 0
    if device:
        loss_fn = loss_fn.to(device)

    for img_file in os.listdir(fake_dir):

        if osp.exists(osp.join(real_dir, img_file)):

            cnt += 1
            img_0 = lpips.im2tensor(lpips.load_image(osp.join(fake_dir, img_file)))
            img_1 = lpips.im2tensor(lpips.load_image(osp.join(real_dir, img_file)))

            if device:
                img_0 = img_0.to(device)
                img_1 = img_1.to(device)
            d = loss_fn.forward(img_0, img_1)
            lpips_dist.append(d.item())

    return sum(lpips_dist) / float(len(lpips_dist)), cnt
