import torch
import os
import os.path as osp
import random
import glob
import pickle
import json
import copy

from torch.utils import data
from torchvision import transforms as T
from PIL import Image

class ErrorDataset(data.Dataset):

    def __init__(self, config, transform_target, transform_source, mode='train'):
        """Initialize and preprocess the Polyevore dataset."""
        self.image_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], mode)
        self.train_dir = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.transform_target = transform_target
        self.transform_source = transform_source
        self.mode = config['TRAINING_CONFIG']['MODE']
        self.seed = config['TRAINING_CONFIG']['SEED']
        self.dataset = list()

        self.outfit_list = glob.glob(osp.join(self.image_dir, '*'))
        random.seed(12345)
        random.shuffle(self.outfit_list)

        for outfit_path in random.sample(self.outfit_list, 30):
            t_idx = random.sample(list(range(1, 6)), 1)[0]
            base_list = [t_idx] * 4
            assert len(base_list) == 4
            self.dataset.append([outfit_path, t_idx, base_list])

        #for outfit_path in random.sample(self.outfit_list, 50):
        #    t_idx_list = random.sample(list(range(1, 6)), 1)
        #    for t_idx in t_idx_list:
        #        base_list = list(range(1, 6))
        #        base_list.remove(t_idx)
        #        random.shuffle(base_list)
        #        assert len(base_list) == 4
        #        self.dataset.append([outfit_path, t_idx, base_list])

        self.dataset.append(['', 0, ['0'] * 4])
        self.dataset.append(['', 1, ['1'] * 4])
        self.dataset.append(['zero', 1, ['1'] * 4])

        with open(osp.join(self.train_dir, f'{mode}_data_list.plk'), 'wb') as fp:
            pickle.dump(self.dataset, fp)
        print('dataset pickle is dump')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        outfit_path, t_idx, base_list = self.dataset[index]
        if outfit_path == 'zero':
            z_mode = outfit_path
            outfit_path = ''
        else:
            z_mode = ''
        source_img_list = list()

        outfit_id = 0

        target_image = Image.open(os.path.join(outfit_path, f'{t_idx}.jpg'))
        target_image = target_image.convert('RGB')

        for s_idx in base_list:
            temp_image = Image.open(os.path.join(outfit_path, f'{s_idx}.jpg'))
            temp_image = self.transform_source(temp_image.convert('RGB'))

            if z_mode == 'zero':
                source_img_list.append(torch.zeros_like(temp_image))
            else:
                source_img_list.append(temp_image)

        source_img_list = torch.stack(source_img_list)

        if z_mode == 'zero':
            return torch.LongTensor([int(outfit_id)]), torch.LongTensor([int(t_idx)]), torch.zeros_like(self.transform_target(target_image)), source_img_list
        else:
            return torch.LongTensor([int(outfit_id)]), torch.LongTensor([int(t_idx)]), self.transform_target(target_image), source_img_list

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)

class CombDataset(data.Dataset):
    """Dataset class for the Polyevore dataset."""

    def __init__(self, config, transform_target, transform_source):
        """Initialize and preprocess the Polyevore dataset."""
        self.transform_target = transform_target
        self.transform_source = transform_source
        self.img_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], 'train')
        self.json_path = config['TRAINING_CONFIG']['COMB_JSON']
        with open(self.json_path, 'r') as fp:
            self.json_data = json.load(fp)
        self.dataset = list()

        for data in self.json_data:
            outfit_id = data['outfit_id']
            r_idx = int(data['idx'])

            t_idx_list = random.sample(list(range(1, 6)), 5)
            for t_idx in t_idx_list:

                if r_idx == t_idx:
                    continue

                base_list = list(range(1, 6))
                base_list.remove(t_idx)
                assert len(base_list) == 4
                index_replace = base_list.index(r_idx)

                for ii, candi_path in enumerate(data['candidate']):
                    base_list_copy = copy.deepcopy(base_list)
                    base_list_copy = [osp.join(self.img_dir, outfit_id, f'{x}.jpg') for x in base_list_copy]
                    base_list_copy[index_replace] = candi_path
                    candi_id = int(candi_path.split(os.sep)[-2])
                    c_idx = int(candi_path.split(os.sep)[-1].replace('.jpg', ''))
                    self.dataset.append([outfit_id, t_idx, base_list_copy, candi_id, c_idx, ii])

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        outfit_path, t_idx, base_list, candi_id, c_idx, ii = self.dataset[index]
        outfit_path = osp.join(self.img_dir, outfit_path)
        source_img_list = list()

        outfit_id = osp.basename(outfit_path)

        target_image = Image.open(os.path.join(outfit_path, f'{t_idx}.jpg'))
        target_image = target_image.convert('RGB')

        for s_path in base_list:
            temp_image = Image.open(s_path)
            temp_image = self.transform_source(temp_image.convert('RGB'))
            source_img_list.append(temp_image)

        source_img_list = torch.stack(source_img_list)

        return torch.LongTensor([int(outfit_id)]), torch.LongTensor([int(t_idx)]), self.transform_target(target_image),\
               source_img_list, torch.LongTensor([int(candi_id)]), torch.LongTensor([int(c_idx)]), torch.LongTensor([int(ii)])

    def __len__(self):
        return len(self.dataset)


class OutfitDataset(data.Dataset):
    """Dataset class for the Polyevore dataset."""

    def __init__(self, config, transform_target, transform_source, mode='train'):
        """Initialize and preprocess the Polyevore dataset."""
        self.image_dir = osp.join(config['TRAINING_CONFIG']['IMG_DIR'], mode)
        self.train_dir = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.transform_target = transform_target
        self.transform_source = transform_source
        self.mode = config['TRAINING_CONFIG']['MODE']
        self.seed = config['TRAINING_CONFIG']['SEED']
        self.data_num = config['TRAINING_CONFIG']['DATA_NUM']

        assert self.data_num in list(range(1, 6))

        self.dataset = list()

        self.outfit_list = glob.glob(osp.join(self.image_dir, '*'))
        random.shuffle(self.outfit_list)

        if mode != 'train':
            self.data_num = 5

        for outfit_path in self.outfit_list:
            t_idx_list = random.sample(list(range(1, 6)), self.data_num)
            for t_idx in t_idx_list:
                base_list = list(range(1, 6))
                base_list.remove(t_idx)
                assert len(base_list) == 4
                self.dataset.append([outfit_path, t_idx, base_list])

        with open(osp.join(self.train_dir, f'{mode}_data_list.plk'), 'wb') as fp:
            pickle.dump(self.dataset, fp)
        print('dataset pickle is dump')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        outfit_path, t_idx, base_list = self.dataset[index]
        source_img_list = list()

        outfit_id = osp.basename(outfit_path)

        target_image = Image.open(os.path.join(outfit_path, f'{t_idx}.jpg'))
        target_image = target_image.convert('RGB')

        for s_idx in base_list:
            temp_image = Image.open(os.path.join(outfit_path, f'{s_idx}.jpg'))
            temp_image = self.transform_source(temp_image.convert('RGB'))
            source_img_list.append(temp_image)

        source_img_list = torch.stack(source_img_list)
        #print(f'source_img_list : {source_img_list.size()}')
        #print(f'type: {type(source_img_list)}') # torch.Size([4, 3, 256, 256])
        return torch.LongTensor([int(outfit_id)]), torch.LongTensor([int(t_idx)]), self.transform_target(target_image), source_img_list

    def __len__(self):
        """Return the number of images."""
        return len(self.dataset)


def get_loader(config, mode='train'):
    """Build and return a data loader."""
    transform_target = list()
    transform_source = list()

    transform_source.append(T.Resize((config['TRAINING_CONFIG']['IMG_SIZE'], config['TRAINING_CONFIG']['IMG_SIZE'])))
    transform_source.append(T.ToTensor())
    transform_source.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform_target.append(T.Resize((config['TRAINING_CONFIG']['IMG_SIZE'], config['TRAINING_CONFIG']['IMG_SIZE'])))
    transform_target.append(T.ToTensor())
    transform_target.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform_source = T.Compose(transform_source)
    transform_target = T.Compose(transform_target)

    if mode == 'comb':
        dataset = CombDataset(config, transform_target, transform_source)
    elif mode == 'error':
        dataset = ErrorDataset(config, transform_target, transform_source)
    else:
        dataset = OutfitDataset(config, transform_target, transform_source, mode)

    if mode == 'train' and config['TRAINING_CONFIG']['BATCH_SIZE'] == 1:
        batch_size = 1
    elif mode == 'train':
        batch_size = config['TRAINING_CONFIG']['BATCH_SIZE']
    else:
        batch_size = 1

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=config['TRAINING_CONFIG']['NUM_WORKER'],
                                  drop_last=True)
    return data_loader
