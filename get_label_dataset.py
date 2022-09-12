import os
import os.path as osp
import torch
import torch.nn as nn
import torchvision.models as models
import pickle


from glob import glob
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
from pprint import pprint


def load_model(gpu):

    model = models.inception_v3(aux_logits=False)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, 7)

    ckpt = './inception_014-98-model.ckpt'
    model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage))
    gpu = torch.device(f'cuda:{gpu}')
    model = model.eval().to(gpu)

    return model, gpu


def get_label_dict():
    model, gpu = load_model('0')

    transform = list()
    transform.append(T.Resize((256, 256)))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    class_list = sorted(['tops', 'bags', 'bottoms', 'dresses', 'earrings', 'shoes', 'eyeglasses'])
    label_dict = {'train' : dict(), 'test' : dict()}

    for img_path in tqdm(glob(osp.join('outfitdata_set3_2341', '*', '*', '*.jpg'))):
        img_name = osp.basename(img_path)
        mode = img_path.split(os.sep)[1]
        outfit_id = img_path.split(os.sep)[-2]
        idx = img_name.replace('.jpg', '')

        if int(idx) not in list(range(1, 6)):
            continue
        else:
            fashion_image = Image.open(img_path).convert('RGB')
            tensor = transform(fashion_image).unsqueeze(0).to(gpu)

            prediction = model(tensor)
            _, pred_idx = torch.max(prediction, 1)
            pred_idx = pred_idx.item()
            category = class_list[pred_idx]

            label_dict[mode][f'{outfit_id}_{idx}'] = {'outfit_id': outfit_id,
                                                'idx': idx,
                                                'category': category,
                                                'cate_idx': pred_idx}


    print(f'label_dict keys : {label_dict.keys()}')
    for mode in label_dict.keys():
        print(f'{mode} : {len(label_dict[mode])}')

    with open('outfitdata_set3_tagged.plk', 'wb') as fp:
        pickle.dump(label_dict, fp)


if __name__ == '__main__':
    get_label_dict()

