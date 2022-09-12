import time
import datetime
import os
import torch
import torch.nn as nn
import os.path as osp
import torch.nn.functional as F
import wandb
import random
import numpy as np

torch.backends.cudnn.benchmark = True

from model import Generator as G_o
from model import Unet as R
from model import Discriminator as D_o
from model import Encoder as E_o
from model import EmbedBlock
from torchvision.models import vgg19_bn
from torchvision.utils import save_image
from data_loader import get_loader
from collections import OrderedDict
from glob import glob

vgg_activation = dict()

# https://discuss.pytorch.org/t/forward-hook-activations-for-loss-computation/142903
def get_activation(name):
    def hook(model, input, output):
        vgg_activation[name] = output #.detach()

    return hook


class Solver(object):

    def __init__(self, config):
        """Initialize configurations."""

        assert torch.cuda.is_available()

        self.wandb = config['TRAINING_CONFIG']['WANDB'] == 'True'
        self.seed = config['TRAINING_CONFIG']['SEED']

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

        assert self.img_size in [128, 256]

        if 'G_VER' in config['TRAINING_CONFIG']:
            self.g_ver = config['TRAINING_CONFIG']['G_VER']
        else:
            self.g_ver = 0

        self.LR = config['TRAINING_CONFIG']['LR']
        self.refine = config['TRAINING_CONFIG']['REFINE'] == 'True'
        self.r_epoch = config['TRAINING_CONFIG']['REFINE_EPOCH']
        self.bilinear = config['TRAINING_CONFIG']['UPSAMPLE'] == 'bilinear'

        # architecture configuration
        self.num_item = config['TRAINING_CONFIG']['NUM_ITEM']
        self.num_target = config['TRAINING_CONFIG']['NUM_TARGET']
        self.num_source = self.num_item - self.num_target

        self.encoder_mode = config['TRAINING_CONFIG']['ENCODER_MODE']
        assert self.encoder_mode in [0, 1, 2]
        self.e_start_ch = config['TRAINING_CONFIG']['E_START_CH']
        self.d_start_ch = config['TRAINING_CONFIG']['D_START_CH']
        self.e_last_ch = config['TRAINING_CONFIG']['E_LAST_CH']

        self.e_layer = 7 if 'E_LAYER' not in config['TRAINING_CONFIG'] else config['TRAINING_CONFIG']['E_LAYER']
        self.g_layer = 7 if 'G_LAYER' not in config['TRAINING_CONFIG'] else config['TRAINING_CONFIG']['G_LAYER']
        self.d_layer = 4 if 'D_LAYER' not in config['TRAINING_CONFIG'] else config['TRAINING_CONFIG']['D_LAYER']

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

        if 'VGG_LAYER' not in config['TRAINING_CONFIG']:
            self.target_layer = ['conv_14']
        else:
            vgg_layer = config['TRAINING_CONFIG']['VGG_LAYER']
            self.target_layer = [f'conv_{x}' for x in vgg_layer.split(',')]
        print(f'vgg layer selected {self.target_layer}')

        self.latest_layer = sorted([int(x.split('_')[-1]) for x in self.target_layer])[-1]

        self.backbone_num_class = config['TRAINING_CONFIG']['BB_CLASS']
        self.backbone_ckpt = config['TRAINING_CONFIG']['BB_CKPT']

        self.use_kl_loss = config['TRAINING_CONFIG']['USE_KL_LOSS'] == 'True'
        self.use_emd = config['TRAINING_CONFIG']['USE_EMD'] == 'True'
        self.use_gt = config['TRAINING_CONFIG']['USE_GT'] == 'True'
        self.use_percep = config['TRAINING_CONFIG']['USE_PERCEP'] == 'True'
        self.use_tv_reg = config['TRAINING_CONFIG']['USE_TV_REG'] == 'True'

        self.epoch         = config['TRAINING_CONFIG']['EPOCH']
        self.g_d_bal       = config['TRAINING_CONFIG']['G_D_BAL']
        self.batch_size    = config['TRAINING_CONFIG']['BATCH_SIZE']

        self.g_lr          = float(config['TRAINING_CONFIG']['G_LR'])
        self.d_lr          = float(config['TRAINING_CONFIG']['D_LR'])
        self.r_lr          = float(config['TRAINING_CONFIG']['R_LR'])

        self.lambda_g_fake = config['TRAINING_CONFIG']['LAMBDA_G_FAKE']
        self.lambda_g_gt = config['TRAINING_CONFIG']['LAMBDA_G_GT']
        self.lambda_g_percep = config['TRAINING_CONFIG']['LAMBDA_G_PERCEP']
        self.lambda_g_tv = config['TRAINING_CONFIG']['LAMBDA_G_TV']
        self.lambda_g_kl = config['TRAINING_CONFIG']['LAMBDA_G_KL']

        self.lambda_d_fake = config['TRAINING_CONFIG']['LAMBDA_D_FAKE']
        self.lambda_d_real = config['TRAINING_CONFIG']['LAMBDA_D_REAL']
        self.lambda_d_gp    = config['TRAINING_CONFIG']['LAMBDA_GP']

        self.d_critic      = config['TRAINING_CONFIG']['D_CRITIC']
        self.g_critic      = config['TRAINING_CONFIG']['G_CRITIC']

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.kl_loss = nn.KLDivLoss('batchmean')

        self.gan_loss = config['TRAINING_CONFIG']['GAN_LOSS']
        assert self.gan_loss in ['lsgan', 'wgan', 'vanilla', 'r1loss']

        self.optim = config['TRAINING_CONFIG']['OPTIM']
        self.beta1 = config['TRAINING_CONFIG']['BETA1']
        self.beta2 = config['TRAINING_CONFIG']['BETA2']

        if self.gan_loss == 'lsgan':
            self.adversarial_loss = torch.nn.MSELoss()
        elif self.gan_loss == 'vanilla':
            self.adversarial_loss = torch.nn.BCELoss()

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
            wandb.init(project='dfd_gan_v0', name=self.train_dir)

        self.build_model()

    def build_model(self):

        self.G = G_o(base_channel=self.encoder_last_ch, g_ver=self.g_ver,
                     LR=self.LR, nlayers=self.g_layer).to(self.gpu)
        self.D = D_o(base_channel=self.d_start_ch, LR=self.LR, nlayers=self.d_layer).to(self.gpu)

        self.encoder_list = None
        self.E = None

        if self.use_emd:
            self.Emd = EmbedBlock(self.encoder_last_ch * self.num_source, LR=self.LR).to(self.gpu)

        if self.encoder_mode == 0: # multi
            self.encoder_list = list()
            for _ in range(self.num_source):
                self.encoder_list.append(E_o(start_channel=self.e_start_ch,
                                             target_channel=self.encoder_last_ch,
                                             LR=self.LR, nlayers=self.e_layer).to(self.gpu))
        elif self.encoder_mode == 1: # single
            self.E = E_o(start_channel=self.e_start_ch,
                         target_channel=self.encoder_last_ch,
                         LR=self.LR, nlayers=self.e_layer).to(self.gpu)
        elif self.encoder_mode == 2: # concat image
            self.E = E_o(img_ch=3 * self.num_source,
                         start_channel=self.e_start_ch,
                         target_channel=self.encoder_last_ch * self.num_source,
                         LR=self.LR, nlayers=self.e_layer).to(self.gpu)

        if self.refine:
            self.R = R(bilinear=self.bilinear).to(self.gpu)

        if self.backbone_ckpt == 'imagenet':
            self.vgg19_bn = vgg19_bn(pretrained=True)
            print(f'imagenet weight is used for training')
        else:
            self.vgg19_bn = vgg19_bn(pretrained=False)
            self.vgg19_bn.classifier[-1] = nn.Linear(self.vgg19_bn.classifier[-1].in_features, self.backbone_num_class)

            state_dict = torch.load(self.backbone_ckpt, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            self.vgg19_bn.load_state_dict(new_state_dict)
            print(f'{self.backbone_ckpt} is used for training')

        self.vgg19_bn = self.vgg19_bn.eval().to(self.gpu)
        self.vgg19_bn = self.vgg19_bn.features[:self.latest_layer + 1]

        for layer in self.target_layer:
            self.vgg19_bn[int(layer.split('_')[-1])].register_forward_hook(get_activation(layer))

        # https://discuss.pytorch.org/t/model-train-and-requires-grad/25845
        for param in self.vgg19_bn.parameters():
            param.requires_grad = False

        g_params = list(self.G.parameters())

        if self.use_emd:
            g_params += list(self.Emd.parameters())

        if self.encoder_mode == 0:
            for encoder in self.encoder_list:
                g_params += list(encoder.parameters())
        else:
            g_params += list(self.E.parameters())

        self.g_optimizer = torch.optim.Adam(g_params, self.g_lr, (self.beta1, self.beta2))
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, (self.beta1, self.beta2))

        if self.refine:
            self.r_optimizer = torch.optim.Adam(self.R.parameters(), self.r_lr, (self.beta1, self.beta2))

        print(f'Use {self.lr_decay_policy} on training.')
        if self.lr_decay_policy == 'LambdaLR':
            self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
            self.d_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
            if self.refine:
                self.r_scheduler = torch.optim.lr_scheduler.LambdaLR(self.r_optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        elif self.lr_decay_policy == 'ExponentialLR':
            self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=0.5)
            self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=0.5)
            if self.refine:
                self.r_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.r_optimizer, gamma=0.5)
        elif self.lr_decay_policy == 'StepLR':
            self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size=100, gamma=0.8)
            self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size=100, gamma=0.8)
            if self.refine:
                self.r_scheduler = torch.optim.lr_scheduler.StepLR(self.r_optimizer, step_size=100, gamma=0.8)
        else:
            self.g_scheduler, self.d_scheduler, self.r_scheduler = None, None, None

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.print_network(self.vgg19_bn, 'vgg19_bn')

        if self.use_emd:
            self.print_network(self.Emd, 'Emd')

        if self.encoder_mode == 0:
            for i, encoder in enumerate(self.encoder_list):
                self.print_network(encoder, f'E_{i}')
        else:
            self.print_network(self.E, 'E')

        if self.refine:
            self.print_network(self.R, 'R')

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

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def r1loss(self, inputs, label=None):
        # non-saturating loss with R1 regularization
        l = -1 if label else 1
        return F.softplus(l*inputs).mean()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.gpu)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

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
        elif self.latent_v == 4:
            return self.get_latent_code_4(source_images)
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
        source_image_list = source_image_list.contiguous().view(b, num_i * ch, h, w)

        """
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
        """
        return source_image_list

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
                #print('z size : ',z.size()) # torch.Size([128, 1, 1])
                latent_code.append(z)
            latent_code_list.append(latent_code)

        if self.concat_mode == 0:
            for b in range(len(latent_code_list)):
                latent_code_list[b] = torch.cat(latent_code_list[b])
        else:
            raise Exception('Unspecified concat mode!')

        latent_code_list = torch.stack(latent_code_list)
        return latent_code_list

    def get_latent_code_4(self, source_image_list):

        return

    def load_model(self):

        ckpt_list = glob(osp.join(self.model_dir, '*.ckpt'))

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

            print(f'All model is laod from {last_ckpt}')

            self.g_optimizer.load_state_dict(ckpt_dict['G_optim'])
            self.d_optimizer.load_state_dict(ckpt_dict['D_optim'])

            return int(osp.basename(last_ckpt).replace('-model.ckpt', ''))

    def train(self):

        # Set data loader.
        data_loader = self.train_loader
        iterations = len(self.train_loader)
        print('iterations : ', iterations)
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        _, _, fixed_target_images, fixed_source_images = next(data_iter)
        fixed_target_images = fixed_target_images.to(self.gpu)
        fixed_source_images = fixed_source_images.to(self.gpu)

        epoch_r = self.load_model()

        start_time = time.time()
        print('Start training...')

        for e in range(epoch_r, self.epoch):

            if self.g_d_bal == e + 1: # and self.gan_loss == 'lsgan'
                self.g_critic = 1
                self.d_critic = 1
                print('training balance is changed equally')

            for i in range(iterations):

                try:
                    _, _, target_images, source_images = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    _, _, target_images, source_images = next(data_iter)

                target_images = target_images.to(self.gpu)
                source_images = source_images.to(self.gpu)
                loss = dict()

                if self.gan_loss == 'r1loss':
                    target_images.requires_grad = True

                if (i + 1) % self.d_critic == 0:

                    latent_code = self.get_latent_code(source_images)

                    if self.use_emd:
                        latent_code = self.Emd(latent_code)
                    fake_images = self.G(latent_code)
                    real_score = self.D(target_images)
                    fake_score = self.D(fake_images.detach())

                    if self.gan_loss in ['lsgan', 'vanilla']:
                        d_loss_real = self.adversarial_loss(real_score, torch.ones_like(real_score))
                        d_loss_fake = self.adversarial_loss(fake_score, torch.zeros_like(fake_score))
                    elif self.gan_loss == 'wgan':
                        d_loss_real = -torch.mean(real_score)
                        d_loss_fake = torch.mean(fake_score)
                    elif self.gan_loss == 'r1loss':
                        d_loss_real = self.r1loss(real_score, True)
                        d_loss_fake = self.r1loss(fake_score, False)
                    else:
                        raise NotImplemented

                    d_loss = self.lambda_d_real * d_loss_real + self.lambda_d_fake * d_loss_fake

                    if self.gan_loss in ['lsgan', 'wgan']:
                        alpha = torch.rand(target_images.size(0), 1, 1, 1).to(self.gpu)
                        x_hat = (alpha * target_images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
                        out_src = self.D(x_hat)
                        d_loss_gp = self.gradient_penalty(out_src, x_hat)
                        d_loss += self.lambda_d_gp * d_loss_gp
                        loss['D/loss_gp'] = self.lambda_d_gp * d_loss_gp.item()

                    if torch.isnan(d_loss):
                        raise Exception('d_loss_fake is nan at Epoch [{}/{}] Iteration [{}/{}]'.format(e + 1, self.epoch, i + 1, iterations))

                    # Backward and optimize.
                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                    # Logging.
                    loss['D/loss_real'] = self.lambda_d_real * d_loss_real.item()
                    loss['D/loss_fake'] = self.lambda_d_fake * d_loss_fake.item()
                    loss['D/d_loss'] = d_loss.item()

                if (i + 1) % self.g_critic == 0:

                    latent_code = self.get_latent_code(source_images)

                    if self.use_emd:
                        latent_code = self.Emd(latent_code)

                    fake_images = self.G(latent_code)
                    fake_score = self.D(fake_images)

                    if self.gan_loss in ['lsgan', 'vanilla']:
                        g_loss_fake = self.adversarial_loss(fake_score, torch.ones_like(fake_score))
                    elif self.gan_loss == 'wgan':
                        g_loss_fake = -torch.mean(fake_score)
                    elif self.gan_loss == 'r1loss':
                        g_loss_fake = self.r1loss(fake_score, True)
                    else:
                        raise NotImplemented

                    g_loss = self.lambda_g_fake * g_loss_fake

                    if self.use_gt:
                        g_loss_gt = self.l1_loss(fake_images, target_images)
                        g_loss += self.lambda_g_gt * g_loss_gt
                        loss['G/loss_gt'] = self.lambda_g_gt * g_loss_gt.item()

                    if self.use_percep:

                        fake_activation, real_activation = dict(), dict()

                        self.vgg19_bn(target_images)
                        for layer in self.target_layer:
                            fake_activation[layer] = vgg_activation[layer]
                        vgg_activation.clear()

                        self.vgg19_bn(fake_images)
                        for layer in self.target_layer:
                            real_activation[layer] = vgg_activation[layer]
                        vgg_activation.clear()

                        g_loss_percep = 0
                        for layer in self.target_layer:
                            g_loss_percep += self.l1_loss(fake_activation[layer], real_activation[layer])

                        g_loss += self.lambda_g_percep * g_loss_percep
                        loss['G/loss_percep'] = self.lambda_g_percep * g_loss_percep.item()

                    if self.use_tv_reg:
                        g_loss_tv = self.lambda_g_tv * (torch.sum(torch.abs(fake_images[:, :, :, :-1] - fake_images[:, :, :, 1:])) +
                                                    torch.sum(torch.abs(fake_images[:, :, :-1, :] - fake_images[:, :, 1:, :])))
                        g_loss += g_loss_tv
                        loss['G/loss_tv'] = g_loss_tv.item()

                    if self.use_kl_loss:
                        g_loss_kl = self.lambda_g_kl * self.kl_loss(latent_code, torch.randn_like(latent_code))
                        g_loss += g_loss_kl
                        loss['G/loss_kl'] = g_loss_kl.item()

                    if torch.isnan(g_loss):
                        raise Exception('d_loss_fake is nan at Epoch [{}/{}] Iteration [{}/{}]'.format(e + 1, self.epoch, i + 1, iterations))

                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = self.lambda_g_fake * g_loss_fake.item()
                    loss['G/g_loss'] = g_loss.item()

                if self.refine and e + 1 >= self.r_epoch:
                    with torch.no_grad():
                        latent_code = self.get_latent_code(source_images)
                        if self.use_emd:
                            latent_code = self.Emd(latent_code)
                        fake_images = self.G(latent_code).detach()
                    refine_images = self.R(fake_images)
                    r_loss = self.l1_loss(refine_images, target_images)

                    self.r_optimizer.zero_grad()
                    r_loss.backward()
                    self.r_optimizer.step()
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
                    latent_code_test = self.get_latent_code(fixed_source_images)
                    if self.use_emd:
                        latent_code_test = self.Emd(latent_code_test)
                    sample_images = self.G(latent_code_test)
                    image_report.append(sample_images)
                    if self.refine and e + 1 >= self.r_epoch:
                        image_report.append(self.R(sample_images))
                    image_report.append(fixed_target_images)
                    for x in range(self.num_source):
                        # print 'fixedSourceImageList[:,x].size() : ',fixedSourceImageList[:,x].size()
                        image_report.append(fixed_source_images[:, x])
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
                ckpt['D'] = self.D.state_dict()
                ckpt['G_optim'] = self.g_optimizer.state_dict()
                ckpt['D_optim'] = self.d_optimizer.state_dict()

                if self.use_emd:
                    ckpt['Emd'] = self.Emd.state_dict()

                if self.encoder_mode == 0:
                    for x in range(len(self.encoder_list)):
                        ckpt[f'E_{x}'] = self.encoder_list[x].state_dict()
                else:
                    ckpt[f'E'] = self.E.state_dict()

                if self.refine and e + 1 >= self.r_epoch:
                    ckpt['R'] = self.R.state_dict()
                    ckpt['R_optim'] = self.r_optimizer.state_dict()

                torch.save(ckpt, ckpt_path)
                print('Saved model checkpoints into {}...'.format(self.model_dir))

            if self.wandb:
                wandb.log({'G/lr': self.g_optimizer.param_groups[0]['lr']})
                wandb.log({'D/lr': self.d_optimizer.param_groups[0]['lr']})

                if self.refine and e + 1 >= self.r_epoch:
                    wandb.log({'R/lr': self.r_optimizer.param_groups[0]['lr']})

            if self.lr_decay_policy != 'None':
                self.g_scheduler.step()
                self.d_scheduler.step()
                if self.refine and e + 1 >= self.r_epoch:
                    self.r_scheduler.step()

        print('Training is finished')

    def test(self, data_loader, epoch, mode='train'):
        z_f = len(str(self.epoch))
        epoch_str = str(epoch).zfill(z_f)
        fake_result_dir1 = osp.join(self.result_dir, f'{mode}_{epoch_str}')
        fake_result_dir2 = osp.join(self.result_dir, f'{mode}_{epoch_str}_fake')
        fake_result_dir3 = osp.join(self.result_dir, f'{mode}_{epoch_str}_refine')
        fake_result_dir4 = osp.join(self.result_dir, f'{mode}_{epoch_str}_paper')

        os.makedirs(fake_result_dir1, exist_ok=True)
        os.makedirs(fake_result_dir2, exist_ok=True)
        os.makedirs(fake_result_dir4, exist_ok=True)

        if self.refine and epoch >= self.r_epoch:
            os.makedirs(fake_result_dir3, exist_ok=True)

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

                if self.refine and epoch >= self.r_epoch:
                    refine_img = self.R(fake_img)
                    image_report.append(refine_img)
                    image_paper.append(refine_img)

                image_report.append(target_images)

                for x in range(self.num_source):
                    image_report.append(source_images[:, x])
                    image_paper.append(source_images[:, x])
                x_report = torch.cat(image_report, dim=3)
                x_paper = torch.cat(image_paper, dim=3)

                sample_path = osp.join(fake_result_dir1, f'{outfit_id}_{str(t_idx)}.jpg') # .zfill(filling)
                save_image(self.denorm(x_report.data.cpu()), sample_path, nrow=1, padding=0)

                sample_path = osp.join(fake_result_dir2, f'{outfit_id}_{str(t_idx)}.jpg') # .zfill(filling)
                save_image(self.denorm(fake_img.data.cpu()), sample_path, nrow=1, padding=0)

                if self.refine and epoch >= self.r_epoch:
                    sample_path = osp.join(fake_result_dir3, f'{outfit_id}_{str(t_idx)}.jpg')# .zfill(filling)
                    save_image(self.denorm(refine_img.data.cpu()), sample_path, nrow=1, padding=0)

                sample_path = osp.join(fake_result_dir4, f'{outfit_id}_{str(t_idx)}.jpg') # .zfill(filling)
                save_image(self.denorm(x_paper.data.cpu()), sample_path, nrow=1, padding=0)