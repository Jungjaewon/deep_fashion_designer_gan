import os.path as osp
import os
import glob
import shutil
import cv2
import numpy as np
import pickle

from tqdm import tqdm
from collections import defaultdict

target_dir = 'outfitdata_set3_2341'

dst_dir = 'test_img'


# get val and test image
def get_test_image():

    os.makedirs(dst_dir, exist_ok=True)

    for mode in ['test']:
        print(f'{mode} is on processing!!')
        dst_img_dir = osp.join(dst_dir, mode)
        os.makedirs(dst_img_dir, exist_ok=True)

        for outfit_path in tqdm(glob.glob(osp.join(target_dir, mode, '*'))):

            outfit_id = osp.basename(outfit_path)
            for image_path in glob.glob(osp.join(target_dir, mode, outfit_id, '*.jpg')):

                img_name = osp.basename(image_path)
                if img_name not in ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']:
                    continue
                else:
                    img_name = f'{outfit_id}_{img_name}'
                    shutil.copy(src=image_path, dst=osp.join(dst_img_dir, img_name))

                    img = cv2.imread(osp.join(dst_img_dir, img_name))
                    img = cv2.resize(img, (256,256))
                    cv2.imwrite(osp.join(dst_img_dir, img_name), img)

def rename_dir():

    #print(glob.glob(osp.join('./exp_*')))
    for exp_dir in glob.glob(osp.join('./exp_*')):
        #print(glob.glob(osp.join(exp_dir, '*')))
        for exp in glob.glob(osp.join(exp_dir, '*')):
            exp_folder = osp.basename(exp)
            dirname = osp.dirname(exp)
            if '_b150_' in exp_folder:
                exp_folder = exp_folder.replace('_b150_', '_')
                #print(f'exp_folder : {exp_folder}')
                os.rename(src=exp, dst=osp.join(dirname, exp_folder))

def delete_outfit():

    add_dir = r'C:\Users\woodc\Desktop\additional_data'
    origin_dir = r'D:\dataset\polyvore_original\data\images'
    second_dir = r'C:\Users\woodc\Desktop\polyvore_second'

    second_outfit_list = glob.glob(osp.join(second_dir, '*'))
    second_outfit_list = [osp.basename(x) for x in second_outfit_list]

    def deletion_dir(outfit_dir):
        del_cnt = 0
        for outfit_path in glob.glob(osp.join(outfit_dir, '*')):
            outfit_id = osp.basename(outfit_path)
            if outfit_id in second_outfit_list:
                shutil.rmtree(outfit_path)
                del_cnt += 1
        print(outfit_dir)
        print(f'del_cnt : {del_cnt}')

    deletion_dir(origin_dir)
    deletion_dir(add_dir)


def del_5_outfit():
    origin_dir = r'D:\dataset\polyvore_original\data\images'

    del_cnt = 0
    for outfit_path in tqdm(glob.glob(osp.join(origin_dir, '*'))):
        if len(glob.glob(osp.join(outfit_path, '*.jpg'))) <= 5:
            shutil.rmtree(outfit_path)
            del_cnt += 1

    print(f'del_cnt : {del_cnt}')


def make_image():

    origin_dir = r'D:\dataset\polyvore_original\data\images'
    target_dir = r'C:\Users\woodc\Desktop\outfit_images'
    n = 10
    for outfit_path in tqdm(glob.glob(osp.join(origin_dir, '*'))):

        outfit_id = osp.basename(outfit_path)
        img_list = glob.glob(osp.join(outfit_path, '*.jpg'))
        img_list.sort(key=lambda x: int(osp.basename(x).replace('.jpg', '')))
        img_list = list(map(cv2.imread,img_list))
        img_list = [cv2.resize(x, (256,256)) for x in img_list]
        img_list = [img_list[i:i + n] for i in range(0, len(img_list), n)]

        for i in range(len(img_list)):
            if len(img_list[i]) < n:
                for j in range(n - len(img_list[i])):
                    img_list[i].append(np.zeros_like(img_list[i][-1]))
                img_list[i] = cv2.hconcat(img_list[i])
            else:
                img_list[i] = cv2.hconcat(img_list[i])

        img_list = cv2.vconcat(img_list)
        cv2.imwrite(osp.join(target_dir, f'{outfit_id}.jpg'), img_list)


def cleaning_data():

    origin_dir = r'C:\Users\woodc\Desktop\outfit_images'
    target_dir = r'C:\Users\woodc\Desktop\filter_done'

    for outfit_path in sorted(glob.glob(osp.join(origin_dir, '*.jpg'))):
        outfit_id = osp.basename(outfit_path)
        img = cv2.imread(outfit_path)
        cv2.namedWindow(outfit_id)        # Create a named window
        cv2.moveWindow(outfit_id, 0,0)  # Move it to (40,30)
        #cv2.resizeWindow(outfit_id, 200, 100)
        cv2.setWindowProperty(outfit_id, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow(outfit_id, img)

        keycode = cv2.waitKey()

        if keycode == ord('q'):
            shutil.copy(outfit_path, osp.join(target_dir, f'{outfit_id}.jpg'))
            os.remove(outfit_path)
        elif keycode == ord('e'):
            os.remove(outfit_path)
        else:
            break

        cv2.destroyAllWindows()

def data_copy():

    filter_dir = r'C:\Users\woodc\Desktop\filter_done'
    origin_dir = r'D:\dataset\polyvore_original\data\images'
    done_dir = r'C:\Users\woodc\Desktop\outfit_data'
    filter_img_list = glob.glob(osp.join(filter_dir, '*.jpg'))
    filter_outfit_list = [osp.basename(x).replace('.jpg', '') for x in filter_img_list]

    for outfit_id in tqdm(filter_outfit_list):

        shutil.copytree(src=osp.join(origin_dir, outfit_id),
                        dst=osp.join(done_dir, outfit_id))

def check_image():

    test_gt_image = glob.glob(osp.join('test_img', 'test', '*.jpg'))
    test_image = glob.glob(osp.join('/home/ubuntu/research_j/deep_fashion_designer_gan/lsgan_g2_glr3e5_dsv3_t2/results/test_150', '*.jpg'))

    test_gt_names = [osp.basename(x) for x in test_gt_image]
    test_image_names = [osp.basename(x) for x in test_image]

    for img_name in test_gt_names:
        assert img_name in test_image_names

    print('Check Done!!')

def get_white_black():

    a = np.zeros((256,256,3))
    cv2.imwrite('black.jpg', a)

    a = np.ones((256, 256, 3)) * 255
    cv2.imwrite('white.jpg', a)


def get_dataset_statistic():

    with open('outfitdata_set3_tagged.plk', 'rb') as fp:
        label_dataset = pickle.load(fp)

    for mode in label_dataset:
        stat_dict = defaultdict(list)
        cate_stat_dict = defaultdict(int)
        outfit_id_set = set()
        print('*' * 10)
        print(f'{mode} dataset statistic')
        print('*' * 10)
        for outfitid_idx in label_dataset[mode]:
            outfit_id, idx = outfitid_idx.split('_')
            outfit_id_set.add(outfit_id)

        for outfit_id in outfit_id_set:
            outfit_type = list()
            for i in range(1,6):
                category = label_dataset[mode][f'{outfit_id}_{i}']['category']
                outfit_type.append(category)
                cate_stat_dict[category] += 1
            stat_dict['_'.join(outfit_type)].append(outfit_id)

        for outfit_type, outfit_ids in stat_dict.items():
            print(f'{outfit_type} : {len(outfit_ids)}')

        for category, cnt in cate_stat_dict.items():
            print(f'{category} : {cnt}')

if __name__ == '__main__':
    pass
    #get_image()
    #rename_dir()
    #delete_outfit()
    #del_5_outfit()
    #make_image()
    #cleaning_data()
    #data_copy()
    #check_image()
    #get_white_black()


    #get_test_image()
    get_dataset_statistic()
