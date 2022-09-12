import os
import torch
import os.path as osp
import torch.backends.cudnn as cudnn
import glob

torch.backends.cudnn.benchmark = True

from model import Generator as G_o
from model import Unet as R
from model import Encoder as E_o
from model import EmbedBlock
from torchvision.utils import save_image
from data_loader import get_loader


class Walker(object):

    def __init__(self, config):
        """Initialize configurations."""

        assert torch.cuda.is_available()

        self.target_dir = config['TRAINING_CONFIG']['TAGET_DIR']

        self.train_loader = get_loader(config, 'train')
        self.val_loader = get_loader(config, 'val')
        self.test_loader = get_loader(config, 'test')
        self.mode = config['TRAINING_CONFIG']['MODE']
        self.img_size    = config['TRAINING_CONFIG']['IMG_SIZE']
        self.attention   = config['TRAINING_CONFIG']['ATTENTION']
        self.block       = config['TRAINING_CONFIG']['BlOCK']

        if 'G_VER' in config['TRAINING_CONFIG']:
            self.g_ver = config['TRAINING_CONFIG']['G_VER']
        else:
            self.g_ver = 0

        self.LR          = config['TRAINING_CONFIG']['LR']
        self.bilinear = config['TRAINING_CONFIG']['UPSAMPLE'] == 'bilinear'

        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']

        if self.attention == '':
            self.attention = None

        print(f'image size is {self.img_size}, block is {self.block} block')
        assert self.img_size in [256] and self.block in ['conv', 'res']

        # architecture configuration
        self.num_item = config['TRAINING_CONFIG']['NUM_ITEM']
        self.num_target = config['TRAINING_CONFIG']['NUM_TARGET']
        self.num_source = self.num_item - self.num_target

        self.encoder_multi = config['TRAINING_CONFIG']['ENCODER_MULTI'] == 'True'
        self.e_start_ch = config['TRAINING_CONFIG']['E_START_CH']
        self.e_last_ch = config['TRAINING_CONFIG']['E_LAST_CH']

        self.idx = config['WALK_CONFIG']['IDX']
        self.step = float(config['WALK_CONFIG']['STEP'])
        self.total_step = config['WALK_CONFIG']['TOTAL_STEP']
        print(f'idx : {self.idx}, step : {self.step}, total_step : {self.total_step}')

        assert self.idx in list(range(self.e_start_ch * self.num_source))

        self.latent_size = config['TRAINING_CONFIG']['LATENT_SIZE']
        self.latent_v    = config['TRAINING_CONFIG']['LATENT_V']

        if self.latent_size == 4:
            self.encoder_last_ch = 256
        elif self.latent_size == 8:
            self.encoder_last_ch = 128
        elif self.latent_size == 16:
            self.encoder_last_ch = 64
        elif self.latent_size == 32:
            self.encoder_last_ch = 32
        elif self.latent_size == 1:
            self.encoder_last_ch = self.e_last_ch
        else:
            raise NotImplemented

        self.concat_mode = config['TRAINING_CONFIG']['CONCAT_MODE']
        self.emd_repeat = config['TRAINING_CONFIG']['EMD_REPEAT']
        self.emd_fnc = config['TRAINING_CONFIG']['EMD_FNC']
        self.use_emd = config['TRAINING_CONFIG']['USE_EMD'] == 'True'
        self.g_spec = config['TRAINING_CONFIG']['G_SPEC'] == 'True'

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.gpu = torch.device(f'cuda:{self.gpu}')

        # Directory
        self.result_dir = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.sample_dir = osp.join(self.result_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])

        self.build_model()

    def build_model(self):

        self.G = G_o(base_channel=self.encoder_last_ch, LR=self.LR, block=self.block,
                     spec=self.g_spec, g_ver=self.g_ver).to(self.gpu)
        self.G.eval()

        if self.use_emd:
            self.Emd = EmbedBlock(self.emd_repeat, self.encoder_last_ch * self.num_source,
                                  LR=self.LR, attention=self.attention, fnc=self.emd_fnc).to(self.gpu)
            self.Emd.eval()

        if self.encoder_multi:
            self.encoder_list = list()
            for _ in range(self.num_source):
                self.encoder_list.append(E_o(start_channel=self.e_start_ch,
                                             target_channel=self.encoder_last_ch,
                                             LR=self.LR, block=self.block,
                                             spec=self.g_spec).to(self.gpu))
                self.encoder_list[-1].eval()
        else:
            self.E = E_o(start_channel=self.e_start_ch,
                         target_channel=self.encoder_last_ch,
                         block = self.block, LR=self.LR, spec=self.g_spec).to(self.gpu)
            self.E.eval()

        #self.R = R(bilinear=self.bilinear).to(self.gpu)
        #self.R.eval()

        self.print_network(self.G, 'G')

        if self.use_emd:
            self.print_network(self.Emd, 'Emd')

        if self.encoder_multi:
            for i, encoder in enumerate(self.encoder_list):
                self.print_network(encoder, f'E_{i}')
        else:
            self.print_network(self.E, 'E')

        #self.print_network(self.R, 'R')

    def load_model(self):
        last_ckpt = sorted(glob.glob(osp.join(self.target_dir, 'models', '*.ckpt')))[-1]

        ckpt_dict = torch.load(last_ckpt)
        print(f'ckpt_dict key : {ckpt_dict.keys()}')
        self.G.load_state_dict(ckpt_dict['G'])
        print('G is load')

        if self.use_emd:
            self.Emd.load_state_dict(ckpt_dict['Emd'])
            print('Emd is load')

        if self.encoder_multi:
            for i in range(self.num_source):
                self.encoder_list[i].load_state_dict(ckpt_dict[f'E_{i}'])
                print(f'E_{i} is load')
        else:
            self.E.load_state_dict(ckpt_dict['E'])
            print('E is load')

        #self.R.load_state_dict(ckpt_dict['R'])

        print(f'All model is laod from {last_ckpt}')

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(osp.join(self.result_dir, 'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)
            print('', file=fp)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def get_latent_code(self, source_image_list):

        if self.latent_v == 0:
            return self.get_latent_code_v0(source_image_list)
        elif self.latent_v == 1:
            return self.get_latent_code_v1(source_image_list)
        elif self.latent_v == 2:
            return self.get_latent_code_v2(source_image_list)
        elif self.latent_v == 3:
            return self.get_latent_code_old(source_image_list)
        else:
            raise NotImplemented

    def get_latent_code_v0(self, source_image_list):

        #print(f'source_image_list : {source_image_list.size()}') # batch, num_item, ch, h ,w
        latent_code_list = list()
        b_split = torch.chunk(source_image_list, self.batch_size, dim=0)
        for source_imgs in b_split:
            #print(source_imgs.size())
            source_imgs = torch.squeeze(source_imgs, dim=0)
            source_imgs = torch.chunk(source_imgs, self.num_item, dim=0)
            latent_code = list()
            for item in source_imgs:
                z = self.E(item)
                #print(f'z : {z.size()}') # 1 dim latent_size, latent_size
                latent_code.append(z)

            latent_code = torch.cat(latent_code, dim=1)
            #print(f'latent_code : {latent_code.size()}') # 1, 1024, 1, 1
            latent_code_list.append(latent_code)

        latent_code_tensor = torch.cat(latent_code_list, dim=0)
        #print(f'latent_code_tensor : {latent_code_tensor.size()}') # 16, 1024, 1, 1

        return latent_code_tensor

    def get_latent_code_v1(self, source_image_list):

        #print(f'source_image_list : {source_image_list.size()}')
        latent_code_list = list()
        b_split = torch.chunk(source_image_list, self.batch_size, dim=0)
        for source_imgs in b_split:
            #print(source_imgs.size())
            source_imgs = torch.squeeze(source_imgs, dim=0)
            #print(source_imgs.size())
            latent_code = self.E(source_imgs)
            num_src, ch, h, w = latent_code.size()
            latent_code = latent_code.contiguous()
            latent_code = latent_code.view(num_src * ch, h, w)
            latent_code_list.append(latent_code)

        latent_code_tensor = torch.stack(latent_code_list, dim=0)
        #print(latent_code_tensor.size())
        #raise NotImplemented

        return latent_code_tensor

    def get_latent_code_v2(self, source_image_list):

        #print(f'source_image_list : {source_image_list.size()}') # batch, num_item, ch, h ,w
        b, num_i, ch, h, w = source_image_list.size()
        source_image_list = source_image_list.contiguous().view(b * num_i, ch, h, w)
        source_image_list = self.E(source_image_list)
        _, ch, h, w = source_image_list.size()
        source_image_list = source_image_list.contiguous().view(b, num_i, ch, h, w)

        latent_code_list = list()
        b_split = torch.chunk(source_image_list, self.batch_size, dim=0)
        for source_imgs in b_split:
            latent_code = torch.squeeze(source_imgs, dim=0)
            num_src, ch, h, w = latent_code.size()
            latent_code = latent_code.contiguous()
            latent_code = latent_code.view(num_src * ch, h, w)
            latent_code_list.append(latent_code)
        latent_code_tensor = torch.stack(latent_code_list, dim=0)
        return latent_code_tensor

    def get_latent_code_old(self, source_image_list):

        latent_code_list = list()

        for b in range(source_image_list.size(0)):
            latent_code = list()
            for i in range(len(source_image_list[b])):
                #https://discuss.pytorch.org/t/what-is-the-difference-between-view-and-unsqueeze/1155
                #https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612
                if self.encoder_multi:
                    z = self.encoder_list[i](source_image_list[b][i].unsqueeze(0))
                else:
                    z = self.E(source_image_list[b][i].unsqueeze(0))
                z = torch.squeeze(z, 0)
                #print('z size : ',z.size()) # torch.Size([128, 4, 4])
                latent_code.append(z)
            latent_code_list.append(latent_code)

        if self.concat_mode == 0:
            for b in range(len(latent_code_list)):
                latent_code_list[b] = torch.cat(latent_code_list[b])
                #print('latent_code_list[b] :', latent_code_list[b].size()) # torch.Size([512, 4, 4])
        else:
            raise Exception('Unspecified concat mode!')

        #assert self.batch_size == len(latent_code_list)
        latent_code_list = torch.stack(latent_code_list)
        return latent_code_list

    def test(self, mode):

        # Set data loader.
        if mode == 'train':
            data_loader = self.train_loader
        elif mode == 'val':
            data_loader = self.val_loader
        elif mode == 'test':
            data_loader = self.test_loader
        else:
            raise NotImplemented

        self.load_model()

        with torch.no_grad():
            for i, data in enumerate(data_loader):

                if i == 20:
                    break

                outfit_id, t_idx, target_images, source_images = data
                outfit_id = int(outfit_id.item())
                t_idx = int(t_idx.item())
                target_images = target_images.to(self.gpu)
                source_images = source_images.to(self.gpu)

                fake_img_list = list()
                refine_img_list = list()
                source_img_list = list()

                image_result_dir = osp.join(self.sample_dir, f'{mode}_{outfit_id}')

                os.makedirs(image_result_dir, exist_ok=True)

                latent_code = self.get_latent_code(source_images)
                if self.use_emd:
                    latent_code = self.Emd(latent_code)

                for s in range(self.total_step + 1):
                    latent_code_new = latent_code.clone()
                    latent_code_new[:, self.idx, :, :] = latent_code_new[:, self.idx, :, :] + self.step * s

                    fake_img = self.G(latent_code_new)
                    fake_img_list.append(fake_img)

                    #refine_img = self.R(fake_img)
                    #refine_img_list.append(refine_img)

                source_img_list.append(target_images)
                for x in range(self.num_source):
                    # print 'fixedSourceImageList[:,x].size() : ',fixedSourceImageList[:,x].size()
                    source_img_list.append(source_images[:, x])
                source_img_list = torch.cat(source_img_list, dim=3)

                fake_img_list = torch.cat(fake_img_list, dim=3)
                #refine_img_list = torch.cat(refine_img_list, dim=3)

                sample_path = osp.join(image_result_dir, f'{outfit_id}_{str(t_idx)}_source.jpg') # .zfill(filling)
                save_image(self.denorm(source_img_list.data.cpu()), sample_path, nrow=1, padding=0)

                sample_path = osp.join(image_result_dir, f'{outfit_id}_{str(t_idx)}_fake.jpg') # .zfill(filling)
                save_image(self.denorm(fake_img_list.data.cpu()), sample_path, nrow=1, padding=0)

                #sample_path = osp.join(image_result_dir, f'{outfit_id}_{str(t_idx)}_refine.jpg')# .zfill(filling)
                #save_image(self.denorm(refine_img_list.data.cpu()), sample_path, nrow=1, padding=0)

