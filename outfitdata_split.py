import glob
import os
import os.path as osp
import random
import shutil
import argparse


def copy_outfit(outfit_path_list, new_data, mode):

    target_dir = osp.join(new_data, mode)
    os.makedirs(target_dir, exist_ok=True)

    print(f'build copy : {mode}')
    for outfit_path in outfit_path_list:

        outfit_id = osp.basename(outfit_path)
        shutil.copytree(outfit_path, osp.join(target_dir, outfit_id))

def data_set():

    outfit_list = glob.glob(osp.join('polyvore_third', '*'))

    fail_out = list()
    for outfit in outfit_list:

        img_list = glob.glob(osp.join(outfit, '*.jpg'))
        img_list = [osp.basename(x) for x in img_list]
        outfit_id = osp.basename(outfit)


        for i in range(1,6):
            if f'{i}.jpg'in img_list:
                pass
            else:
                fail_out.append(outfit_id)
                break

    print(f'fail_outfit : {fail_out}')
    print(f'len(fail_out) : {len(fail_out)}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--new_data', type=str, default='outfitdata_set', help='new data directory')
    config = parser.parse_args()

    data_set()

    outfit_list = glob.glob(osp.join('polyvore_third', '*'))

    if config.seed != 0:
        random.seed(config.seed)
        random.shuffle(outfit_list)

    new_data = f'{config.new_data}_{config.seed}'
    train_cut = int(len(outfit_list) * 0.9)

    train_outfit = outfit_list[:train_cut]
    test_outfit = outfit_list[train_cut:]

    copy_outfit(train_outfit, new_data, 'train')
    copy_outfit(test_outfit, new_data, 'test')
