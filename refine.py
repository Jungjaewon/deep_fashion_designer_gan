import time
import datetime
import os
import torch
import torch.nn as nn
import os.path as osp
import glob
import wandb
import random
import numpy as np

torch.backends.cudnn.benchmark = True

from model import Generator as G_o
from model import Unet as R
from model import Encoder as E_o
from model import EmbedBlock
from model import Discriminator
from torchvision.utils import save_image
from data_loader import get_loader


class Refiner(object):

    def __init__(self, config):
        """Initialize configurations."""

        assert torch.cuda.is_available()

        self.wandb = config['TRAINING_CONFIG']['WANDB'] == 'True'
        self.seed = config['TRAINING_CONFIG']['SEED']
        self.target_dir = config['TRAINING_CONFIG']['TAGET_DIR']

        if self.seed != 0:
            print(f'set seed : {self.seed}')
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # https://hoya012.github.io/blog/reproducible_pytorch/
        else:
            print('do not set seed')

        self.train_loader = get_loader(config, 'train')
        self.test_loader = get_loader(config, 'test')
        self.img_size    = config['TRAINING_CONFIG']['IMG_SIZE']

        if 'G_VER' in config['TRAINING_CONFIG']:
            self.g_ver = config['TRAINING_CONFIG']['G_VER']
        else:
            self.g_ver = 0

        self.LR          = config['TRAINING_CONFIG']['LR']
        self.bilinear = config['TRAINING_CONFIG']['UPSAMPLE'] == 'bilinear'

        # architecture configuration
        self.num_item = config['TRAINING_CONFIG']['NUM_ITEM']
        self.num_target = config['TRAINING_CONFIG']['NUM_TARGET']
        self.num_source = self.num_item - self.num_target

        self.encoder_mode = config['TRAINING_CONFIG']['ENCODER_MODE']
        self.e_start_ch = config['TRAINING_CONFIG']['E_START_CH']
        self.e_last_ch = config['TRAINING_CONFIG']['E_LAST_CH']

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
        self.use_emd = config['TRAINING_CONFIG']['USE_EMD'] == 'True'
        self.use_tv_loss = config['TRAINING_CONFIG']['USE_TV_LOSS'] == 'True'

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']

        self.r_lr          = float(config['TRAINING_CONFIG']['R_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])

        self.loss_use = config['TRAINING_CONFIG']['LOSS_USE']

        if self.loss_use == 'l1':
            self.loss = nn.L1Loss()
        elif self.loss_use == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplemented

        self.adversarial_loss = nn.MSELoss()

        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_r_tv = config['TRAINING_CONFIG']['LAMBDA_R_TV']
        self.lambda_r_fake = config['TRAINING_CONFIG']['LAMBDA_R_FAKE']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        self.gpu = config['TRAINING_CONFIG']['GPU']
        self.gpu = torch.device(f'cuda:{self.gpu}')

        # Directory
        self.train_dir  = config['TRAINING_CONFIG']['TRAIN_DIR']
        self.log_dir    = osp.join(self.train_dir, config['TRAINING_CONFIG']['LOG_DIR'])
        self.sample_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['SAMPLE_DIR'])
        self.result_dir = osp.join(self.train_dir, config['TRAINING_CONFIG']['RESULT_DIR'])
        self.model_dir  = osp.join(self.train_dir, config['TRAINING_CONFIG']['MODEL_DIR'])

        # Steps
        self.log_step       = config['TRAINING_CONFIG']['LOG_STEP']
        self.sample_step    = config['TRAINING_CONFIG']['SAMPLE_STEP']
        self.test_step      = config['TRAINING_CONFIG']['TEST_STEP']
        self.save_step      = config['TRAINING_CONFIG']['SAVE_STEP']
        self.save_start     = config['TRAINING_CONFIG']['SAVE_START']

        self.lr_decay_policy = config['TRAINING_CONFIG']['LR_DECAY_POLICY']
        print(f'lr_decay_policy : {self.lr_decay_policy}')

        if self.wandb:
            wandb.login(key='3b3fd7ec86b8f3f0f32f2d7a78456686d8755d99')
            wandb.init(project='dfd_gan_training', name=self.train_dir)

        self.build_model()


    def build_model(self):

        self.G = G_o(base_channel=self.encoder_last_ch, g_ver=self.g_ver, LR=self.LR).eval().to(self.gpu)

        if self.use_emd:
            self.Emd = EmbedBlock(self.encoder_last_ch * self.num_source, LR=self.LR).eval().to(self.gpu)

        if self.encoder_mode == 0: # multi
            self.encoder_list = list()
            for _ in range(self.num_source):
                self.encoder_list.append(E_o(start_channel=self.e_start_ch,
                                             target_channel=self.encoder_last_ch,
                                             LR=self.LR).eval().to(self.gpu))
                self.encoder_list[-1].eval()
        elif self.encoder_mode == 1: # single
            self.E = E_o(start_channel=self.e_start_ch,
                         target_channel=self.encoder_last_ch,
                         LR=self.LR).eval().to(self.gpu)
        elif self.encoder_mode == 2: # concat image
            self.E = E_o(img_ch=3 * self.num_source,
                         start_channel=self.e_start_ch,
                         target_channel=self.encoder_last_ch * self.num_source,
                         LR=self.LR).eval().to(self.gpu)

        self.R = R(bilinear=self.bilinear).to(self.gpu)
        self.r_optimizer = torch.optim.Adam(self.R.parameters(), self.r_lr, (self.beta1, self.beta2))

        self.D = Discriminator().to(self.gpu)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        print(f'Use {self.lr_decay_policy} on training.')
        if self.lr_decay_policy == 'LambdaLR':
            self.r_scheduler = torch.optim.lr_scheduler.LambdaLR(self.r_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        elif self.lr_decay_policy == 'ExponentialLR':
            self.r_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.r_optimizer, gamma=0.5)
        elif self.lr_decay_policy == 'StepLR':
            self.r_scheduler = torch.optim.lr_scheduler.StepLR(self.r_optimizer, step_size=100, gamma=0.8)
        else:
            self.r_scheduler = None

        self.print_network(self.G, 'G')

        if self.use_emd:
            self.print_network(self.Emd, 'Emd')

        if self.encoder_mode == 0:
            for i, encoder in enumerate(self.encoder_list):
                self.print_network(encoder, f'E_{i}')
        else:
            self.print_network(self.E, 'E')

        self.print_network(self.R, 'R')

    def load_model_gen(self):
        last_ckpt = sorted(glob.glob(osp.join(self.target_dir, 'models', '*.ckpt')),
                           key=lambda x: int(x.split(os.sep)[-1].replace('-model.ckpt', '')))[-1]

        ckpt_dict = torch.load(last_ckpt)
        print(f'ckpt_dict key : {ckpt_dict.keys()}')
        self.G.load_state_dict(ckpt_dict['G'])
        print('G is load')

        if self.use_emd:
            self.Emd.load_state_dict(ckpt_dict['Emd'])
            print('Emd is load')

        if self.encoder_mode == 0:
            for i in range(self.num_source):
                self.encoder_list[i].load_state_dict(ckpt_dict[f'E_{i}'])
                print(f'E_{i} is load')
        else:
            self.E.load_state_dict(ckpt_dict['E'])
            print('E is load')

        print(f'All model is laod from {last_ckpt}')

    def load_model_refine(self):

        ckpt_list = glob.glob(osp.join(self.model_dir, '*.ckpt'))

        if len(ckpt_list) == 0:
            return 0
        else:
            last_ckpt = sorted(ckpt_list,
                               key=lambda x: int(x.split(os.sep)[-1].replace('-model.ckpt', '')))[-1]

            ckpt_dict = torch.load(last_ckpt)
            print(f'ckpt_dict key : {ckpt_dict.keys()}')
            self.G.load_state_dict(ckpt_dict['G'])
            print('G is load')

            if self.use_emd:
                self.Emd.load_state_dict(ckpt_dict['Emd'])
                print('Emd is load')

            if self.encoder_mode == 0:
                for i in range(self.num_source):
                    self.encoder_list[i].load_state_dict(ckpt_dict[f'E_{i}'])
                    print(f'E_{i} is load')
            else:
                self.E.load_state_dict(ckpt_dict['E'])
                print('E is load')

            self.R.load_state_dict(ckpt_dict['R'])
            self.r_optimizer.load_state_dict(ckpt_dict['R_optim'])
            print('R is load')

            self.D.load_state_dict(ckpt_dict['D_r'])
            self.d_optimizer.load_state_dict(ckpt_dict['D_r_optim'])
            print('D_r is load')

            print(f'All model is laod from refine ckpt, {last_ckpt},')

            return int(osp.basename(last_ckpt).replace('-model.ckpt', ''))


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        #print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

        with open(osp.join(self.train_dir,'model_arch.txt'), 'a') as fp:
            print(model, file=fp)
            print(name, file=fp)
            print("The number of parameters: {}".format(num_params),file=fp)
            print('', file=fp)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def get_latent_code(self, source_images):

        if self.encoder_mode == 2:
            b, n, ch, h, w = source_images.size()
            in_source = torch.reshape(source_images, (b, n * ch, h, w))
            return self.E(in_source)

        if self.latent_v == 0:
            return self.get_latent_code_v0(source_images)
        elif self.latent_v == 1:
            return self.get_latent_code_v1(source_images)
        elif self.latent_v == 2:
            return self.get_latent_code_v2(source_images)
        elif self.latent_v == 3:
            return self.get_latent_code_old(source_images)
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
                if self.encoder_mode == 0:
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

    def train(self):

        # Set data loader.
        data_loader = self.train_loader
        iterations = len(self.train_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        _, _, fixed_target_images, fixed_source_iamges = next(data_iter)
        fixed_target_images = fixed_target_images.to(self.gpu)
        fixed_source_iamges = fixed_source_iamges.to(self.gpu)

        self.load_model_gen()
        epoch_r = self.load_model_refine()

        start_time = time.time()
        print('Start training...')

        for e in range(epoch_r, self.epoch):

            for i in range(iterations):

                try:
                    _, _, target_images, source_iamges = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, _, target_images, source_iamges = next(data_iter)

                target_images = target_images.to(self.gpu)
                source_iamges = source_iamges.to(self.gpu)
                loss = dict()

                ####################################################
                # Train the Discriminator
                ####################################################

                with torch.no_grad():
                    latent_code = self.get_latent_code(source_iamges)
                    if self.use_emd:
                        latent_code = self.Emd(latent_code)
                    fake_images = self.G(latent_code)

                real_score = self.D(target_images)
                fake_score = self.D(fake_images.detach())

                d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
                d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))

                d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                loss['D/d_loss'] = d_loss.item()

                ####################################################
                # Train the Generator
                ####################################################

                with torch.no_grad():
                    latent_code = self.get_latent_code(source_iamges)
                    if self.use_emd:
                        latent_code = self.Emd(latent_code)
                    fake_images = self.G(latent_code)

                refine_images = self.R(fake_images)
                fake_score = self.D(refine_images)

                recon_loss = self.loss(refine_images, target_images)

                if self.loss_use == 'l2':
                    recon_loss = recon_loss * 0.5

                gan_loss = self.lambda_r_fake * self.adversarial_loss(fake_score, torch.ones_like(fake_score))

                r_loss = recon_loss + gan_loss

                if self.use_tv_loss:
                    r_loss_tv = self.lambda_r_tv * (
                            torch.sum(torch.abs(refine_images[:, :, :, :-1] - refine_images[:, :, :, 1:])) +
                            torch.sum(torch.abs(refine_images[:, :, :-1, :] - refine_images[:, :, 1:, :])))
                    r_loss += r_loss_tv
                    loss['R/loss_tv'] = self.lambda_r_tv * r_loss_tv.item()

                self.r_optimizer.zero_grad()
                r_loss.backward()
                self.r_optimizer.step()
                loss['R/recon_loss'] = recon_loss.item()
                loss['R/gan_loss'] = gan_loss.item()
                loss['R/refine_loss'] = r_loss.item()

                if self.wandb:
                    for tag, value in loss.items():
                        wandb.log({tag: value})

                if (i + 1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Epoch [{}/{}], Elapsed [{}], Iteration [{}/{}]".format(e+1, self.epoch, et, i + 1, iterations)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

            if (e + 1) % self.sample_step == 0:
                with torch.no_grad():
                    image_report = list()
                    latent_code_test = self.get_latent_code(fixed_source_iamges)
                    if self.use_emd:
                        latent_code_test = self.Emd(latent_code_test)
                    sample_images = self.G(latent_code_test)
                    image_report.append(sample_images)
                    image_report.append(self.R(sample_images))
                    image_report.append(fixed_target_images)
                    for x in range(self.num_source):
                        # print 'fixedSourceImageList[:,x].size() : ',fixedSourceImageList[:,x].size()
                        image_report.append(fixed_source_iamges[:, x])
                    x_concat = torch.cat(image_report, dim=3)
                    # https://stackoverflow.com/questions/134934/display-number-with-leading-zeros/33860138
                    sample_path = osp.join(self.sample_dir, '{}-images.jpg'.format(str(e + 1).zfill(len(str(self.epoch)))))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # test step
            if (e + 1) % self.test_step == 0:
                self.test(self.test_loader, e + 1, 'test')

            # Save model checkpoints.
            if (e + 1) % self.save_step == 0 and (e + 1) >= self.save_start:

                e_len = len(str(self.epoch))
                epoch_str = str(e + 1).zfill(e_len)
                ckpt_path = osp.join(self.model_dir, '{}-model.ckpt'.format(epoch_str))
                ckpt = dict()

                ckpt['G'] = self.G.state_dict()

                if self.use_emd:
                    ckpt['Emd'] = self.Emd.state_dict()

                if self.encoder_mode == 0:
                    for x in range(len(self.encoder_list)):
                        ckpt[f'E_{x}'] = self.encoder_list[x].state_dict()
                else:
                    ckpt[f'E'] = self.E.state_dict()

                ckpt['R'] = self.R.state_dict()
                ckpt['R_optim'] = self.r_optimizer.state_dict()

                ckpt['D_r'] = self.D.state_dict()
                ckpt['D_r_optim'] = self.d_optimizer.state_dict()

                torch.save(ckpt, ckpt_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

            if self.wandb:
                wandb.log({'R/lr': self.r_optimizer.param_groups[0]['lr']})

            if self.lr_decay_policy != 'None':
                self.r_scheduler.step()

        print('Training is finished')

    def test(self, data_loader, epoch, mode='train'):
        # Set data loader.
        # filling = len(str(len(data_loader)))

        z_f = len(str(self.epoch))
        epoch_str = str(epoch).zfill(z_f)
        fake_result_dir1 = osp.join(self.result_dir, f'{mode}_{epoch_str}')
        fake_result_dir2 = osp.join(self.result_dir, f'{mode}_{epoch_str}_fake')
        fake_result_dir3 = osp.join(self.result_dir, f'{mode}_{epoch_str}_refine')
        fake_result_dir4 = osp.join(self.result_dir, f'{mode}_{epoch_str}_paper')

        os.makedirs(fake_result_dir1, exist_ok=True)
        os.makedirs(fake_result_dir2, exist_ok=True)
        os.makedirs(fake_result_dir3, exist_ok=True)
        os.makedirs(fake_result_dir4, exist_ok=True)

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                outfit_id, t_idx, target_images, source_images = data
                outfit_id = int(outfit_id.item())
                t_idx = int(t_idx.item())
                target_images = target_images.to(self.gpu)
                source_images = source_images.to(self.gpu)
                image_report = list()
                image_paper = list()
                latent_code = self.get_latent_code(source_images)
                if self.use_emd:
                    latent_code = self.Emd(latent_code)

                fake_img = self.G(latent_code)
                image_report.append(fake_img)
                image_paper.append(fake_img)

                refine_img = self.R(fake_img)
                image_report.append(refine_img)
                image_paper.append(refine_img)

                image_report.append(target_images)

                for x in range(self.num_source):
                    # print 'fixedSourceImageList[:,x].size() : ',fixedSourceImageList[:,x].size()
                    image_report.append(source_images[:, x])
                    image_paper.append(source_images[:, x])
                x_report = torch.cat(image_report, dim=3)
                x_paper = torch.cat(image_paper, dim=3)

                sample_path = osp.join(fake_result_dir1, f'{outfit_id}_{str(t_idx)}.jpg') # .zfill(filling)
                save_image(self.denorm(x_report.data.cpu()), sample_path, nrow=1, padding=0)

                sample_path = osp.join(fake_result_dir2, f'{outfit_id}_{str(t_idx)}.jpg') # .zfill(filling)
                save_image(self.denorm(fake_img.data.cpu()), sample_path, nrow=1, padding=0)

                sample_path = osp.join(fake_result_dir3, f'{outfit_id}_{str(t_idx)}.jpg')# .zfill(filling)
                save_image(self.denorm(refine_img.data.cpu()), sample_path, nrow=1, padding=0)

                sample_path = osp.join(fake_result_dir4, f'{outfit_id}_{str(t_idx)}.jpg') # .zfill(filling)
                save_image(self.denorm(x_paper.data.cpu()), sample_path, nrow=1, padding=0)