import glob
import os.path as osp
import os
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import pickle
import random
import cv2
import shutil
import json
import copy

from numpy import dot
from tqdm import tqdm
from numpy.linalg import norm
from torchvision import transforms as T
from PIL import Image
from bit_models import KNOWN_MODELS
from pprint import pprint
from sklearn.preprocessing import normalize
from finch import FINCH

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def load_model(gpu):

    model = models.inception_v3(aux_logits=False)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 7)

    #ckpt = '/home/jongyoul/jaewon/classification_dfd_gan/inception_759_1/models/014-98-model.ckpt'
    ckpt = 'inception_014-98-model.ckpt'
    model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage))
    gpu = torch.device(f'cuda:{gpu}')
    model = model.eval().to(gpu)

    return model, gpu


def load_bit(gpu):
    gpu = torch.device(f'cuda:{gpu}')
    model = KNOWN_MODELS['BiT-M-R101x1'](head_size=1000, zero_head=True)
    model.load_from(np.load(f'BiT-M-R101x1.npz'))
    model = model.eval().to(gpu)
    return model, gpu


def get_category(model, device, transform, img_path):
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    pred = model(tensor)
    _, pred_result = torch.max(pred, 1)
    category = pred_result.item()
    return category


def get_feature(model, device, transform, img_path):
    image = Image.open(img_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    feature = model(tensor)#.squeeze()
    return feature


def processing(data_dir, mode, gpu='0'):

    print(f'{data_dir}, {mode} processing')
    assert osp.exists(data_dir)

    category_dict = dict()
    class_list = sorted(['tops', 'bags', 'bottoms', 'dresses', 'earrings', 'shoes', 'eyeglasses'])

    img_list = glob.glob(osp.join(data_dir, mode, '*', '*.jpg'))
    img_list = [x for x in img_list if int(osp.basename(x).replace('.jpg', '')) in list(range(1, 6))]
    print(f'the number of images : {len(img_list)}')

    transform = list()
    transform.append(T.Resize((256, 256)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    inception, device = load_model(gpu)

    category_dict_path = f'cate_dict_{mode}.plk'
    if osp.exists(category_dict_path):
        with open(category_dict_path, 'rb') as fp:
            category_dict = pickle.load(fp)
    else:
        # classifying images
        for img_path in tqdm(img_list):
            category = get_category(inception, device, transform, img_path)

            if category not in category_dict:
                category_dict[category] = list()
            category_dict[category].append(img_path)

        with open(f'cate_dict_{mode}.plk', 'wb') as fp:
            pickle.dump(category_dict, fp)

    # building a json
    data_list = list()
    db_dict = dict()

    # load bit model
    bit, device = load_bit(gpu)

    for img_path in tqdm(img_list):
        img_index = osp.basename(img_path).replace('.jpg', '')
        outfit_id = img_path.split(os.sep)[-2]

        category = get_category(inception, device, transform, img_path)
        q_feature = get_feature(bit, device, transform, img_path)
        q_feature = q_feature.cpu().detach().numpy()
        #print(f'q_feature : {np.shape(q_feature)}') # (2048)

        if category not in db_dict:
            db_features = list()
            for db_image_path in category_dict[category]:
                feature = get_feature(bit, device, transform, db_image_path)
                feature = feature.cpu().detach().numpy()
                db_features.append([feature, db_image_path])
            db_dict[category] = np.array(db_features, dtype=object)
            print(f'db_dict[{category}] : {len(db_dict[category])}')

        dist_list = list()

        for db_feature, db_image_path in db_dict[category]:
            dist_list.append([cos_sim(np.squeeze(normalize(q_feature)), np.squeeze(normalize(db_feature))), db_image_path])

        dist_list = np.array(dist_list)
        dist_list = np.sort(dist_list, axis=0)[::-1]
        dist_list = dist_list[:20]

        data_list.append({
                        'outfit_id': outfit_id,
                        'img_index': img_index,
                        'base_img_path': img_path,
                        'cate_idx': category,
                        'category': class_list[category],
                        'close_list': dist_list})

    if len(data_list):
        with open(f'emd_space_anal_{mode}.plk', 'wb') as fp:
            pickle.dump(data_list, fp)


def finch_clustering(data_dir, mode, gpu='1'):

    print(f'{data_dir}, {mode} processing')
    assert osp.exists(data_dir)

    category_dict = dict()
    class_list = sorted(['tops', 'bags', 'bottoms', 'dresses', 'earrings', 'shoes', 'eyeglasses'])

    img_list = glob.glob(osp.join(data_dir, mode, '*', '*.jpg'))
    img_list = [x for x in img_list if int(osp.basename(x).replace('.jpg', '')) in list(range(1, 6))]
    print(f'the number of images : {len(img_list)}')

    transform = list()
    transform.append(T.Resize((256, 256)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    inception, device = load_model(gpu)

    category_dict_path = f'cate_dict_{mode}.plk'
    if osp.exists(category_dict_path):
        with open(category_dict_path, 'rb') as fp:
            category_dict = pickle.load(fp)
    else:
        # classifying images
        for img_path in tqdm(img_list):
            category = get_category(inception, device, transform, img_path)

            if category not in category_dict:
                category_dict[category] = list()
            category_dict[category].append(img_path)

        with open(f'cate_dict_{mode}.plk', 'wb') as fp:
            pickle.dump(category_dict, fp)

    # load bit model
    bit, device = load_bit(gpu)

    finch_result = dict()

    for category in tqdm(category_dict):

        cate_feat_list = list()

        img_list = category_dict[category]
        for cate_img_path in img_list:

            feature = get_feature(bit, device, transform, cate_img_path)
            feature = feature.cpu().detach().numpy()
            cate_feat_list.append(feature)

        cate_feat_np = np.concatenate(cate_feat_list, axis=0)
        print(f'cate_feat_np : {np.shape(cate_feat_np)}')
        c, num_clust, req_c = FINCH(cate_feat_np, initial_rank=None, req_clust=None, distance='cosine', verbose=False)
        print(f'category : {class_list[category]}, num_clust : {num_clust}')

        #print(f'type : {type(c)}')
        #print(f'shape : {np.shape(c)}')

        for f_p in range(len(num_clust)):

            img_path2part = dict()
            for idx, part in enumerate(c[:, f_p]):
                fimg_path = img_list[idx]
                img_path2part[fimg_path] = part

            finch_result[f'{class_list[category]}_{f_p}'] = {'cluster' : img_path2part, 'num_clust': num_clust, 'partition': f_p}

    with open(f'finch_{mode}_result.plk', 'wb') as fp:
        pickle.dump(finch_result, fp)


def visualize(mode='train'):

    random.seed(134)

    with open(f'emd_space_anal_{mode}.plk', 'rb') as fp:
        data_list = pickle.load(fp)

    random.shuffle(data_list)

    result_dir = f'emd_{mode}'

    if not osp.exists(result_dir):
        os.mkdir(result_dir)

    for idx, data in enumerate(data_list[:50]):
        print(idx)
        pprint(data)
        base_img_path = data['base_img_path']
        base_img = cv2.imread(base_img_path)
        base_img = cv2.resize(base_img, (256, 256))
        img_result = list()
        img_result.append(base_img)
        for _, img_path in data['close_list']:

            close_img = cv2.imread(img_path)
            close_img = cv2.resize(close_img, (256, 256))
            img_result.append(close_img)

        img_result = cv2.hconcat(img_result)

        cv2.imwrite(osp.join(result_dir, f'{idx}_result_{data["category"]}.jpg'), img_result)


def vis_finch(mode='train'):

    with open(f'finch_{mode}_result.plk', 'rb') as fp:
        finch_result = pickle.load(fp)

    result_dir = f'finch_result_{mode}'

    if not osp.exists(result_dir):
        os.mkdir(result_dir)

    #finch_result[f'{category}_{f_p}'] = {'cluster' : img_path2part, 'num_clust': num_clust, 'partition': f_p}
    part2imglist = dict()
    for cate_fp in tqdm(finch_result):

        if cate_fp.split('_')[-1] != '0':
            continue

        cate_dir = osp.join(result_dir, cate_fp)
        if not osp.exists(cate_dir):
            os.mkdir(cate_dir)

        imgpath2part = finch_result[cate_fp]['cluster']

        for idx, img_path in enumerate(imgpath2part):
            part = imgpath2part[img_path]
            part_dir = osp.join(cate_dir, str(part))
            if not osp.exists(part_dir):
                os.mkdir(part_dir)
            shutil.copy(img_path, osp.join(part_dir, f'{idx}.jpg'))

            k = f'{cate_fp}_{str(part)}'
            if k not in part2imglist:
                part2imglist[k] = list()
            part2imglist[k].append(img_path)

    data_list = list()

    for idx, key in enumerate(part2imglist):

        img_list = part2imglist[key]

        for i in range(len(img_list)):
            img_path = img_list[i]
            category = key.split('_')[0]
            part = key.split('_')[-1]
            img_list_copy = copy.deepcopy(img_list)
            img_list_copy.remove(img_path)
            outfit_id = img_path.split(os.sep)[-2]
            idx = img_path.split(os.sep)[-1].replace('.jpg', '')
            data_list.append({
                'category': category,
                'part': part,
                'img_path': img_path,
                'outfit_id': outfit_id,
                'idx': idx,
                'candidate': img_list_copy
            })

    print(f'len(data_list) : {len(data_list)}')

    with open(f'finch_trans_{mode}.json','w') as fp:
        json.dump(data_list, fp, indent=1)


if __name__ == '__main__':
    for mode in ['train']:
        finch_clustering(data_dir='outfitdata_set3_2341', mode=mode, gpu='0')
        vis_finch(mode='train')