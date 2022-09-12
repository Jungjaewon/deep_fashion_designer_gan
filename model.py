import numpy as np
from scipy.signal import gaussian
from unet_parts import *

"""
Original conv size is 4
but kernel size is reduced to 3....
W = (W - F + 2P) /S + 1
"""

# Reference : https://github.com/milesial/Pytorch-UNet


class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class EmbedBlock(nn.Module):

    def __init__(self, input_channel, num_layer=6, LR=0.01):
        super(EmbedBlock,self).__init__()
        self.num_layer = num_layer
        self.layers = list()

        for _ in range(num_layer):
            self.layers.append(nn.Linear(input_channel, input_channel))
            self.layers.append(nn.LeakyReLU(LR, inplace=True))
        self.main = nn.Sequential(*self.layers)

    def forward(self, x):
        b, ch, _, _ = x.size()
        x = x.contiguous().view(b, ch)
        output = self.main(x)
        output = output.unsqueeze(len(output.size())).unsqueeze(len(output.size()))
        return output


class Encoder(nn.Module):
    """Encoder for translating an image to a latent value, W = (W - F + 2P) /S + 1"""
    def __init__(self, img_ch=3, start_channel=16, target_channel=128, nlayers=7, LR=0.01, last_layer='max'):
        super(Encoder, self).__init__()
        layers = list()

        self.channel = start_channel
        # 256 -> 128
        layers.append(nn.Conv2d(img_ch, self.channel, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(self.channel, affine=True))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        for _ in range(nlayers - 3): # 64 32 16 8
            layers.append(nn.Conv2d(self.channel, self.channel * 2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(self.channel * 2, affine=True))
            layers.append(nn.LeakyReLU(LR, inplace=True))
            self.channel *= 2

        # 8 -> 4
        layers.append(nn.Conv2d(self.channel, target_channel, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(target_channel, affine=True))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        if last_layer == 'max':
            layers.append(nn.MaxPool2d(kernel_size=4))
        elif last_layer == 'avg':
            layers.append(nn.AvgPool2d(kernel_size=4))
        elif last_layer == 'conv':
            layers.append(nn.Conv2d(target_channel, target_channel, kernel_size=4, stride=1, bias=False))
            layers.append(nn.LeakyReLU(LR, inplace=True))
        else:
            raise NotImplemented

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Generator(nn.Module):
    """Generator network. Conv : W = (W - F + 2P) /S + 1 / TransPosed : W = (Win - 1) * S - 2P + F + OutP"""
    def __init__(self, base_channel=128, nlayers=7, n_item=4, LR=0.01, g_ver=0):
        super(Generator, self).__init__()

        layers = list()
        self.n_item = n_item
        self.g_ver = g_ver
        self.mchannel = base_channel * self.n_item * 4
        self.startChannel = base_channel * self.n_item

        # size 1 to 4
        layers.append(nn.ConvTranspose2d(self.startChannel, self.mchannel, kernel_size=4, bias=False))
        layers.append(nn.InstanceNorm2d(self.mchannel, affine=True))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        for _ in range(nlayers - 2): # 8 16 32 64 128
            if self.g_ver == 0:
                layers.append(nn.ConvTranspose2d(self.mchannel, self.mchannel // 2, kernel_size=4, stride=2, padding=1, bias=False))
            elif self.g_ver == 1:
                layers.append(nn.ConvTranspose2d(self.mchannel, self.mchannel // 2, kernel_size=2, stride=2, bias=False))
            elif self.g_ver == 2:
                layers.append(nn.ConvTranspose2d(self.mchannel, self.mchannel // 2, kernel_size=4, stride=2, padding=1, bias=False))
                layers.append(nn.Conv2d(self.mchannel // 2, self.mchannel // 2, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(self.mchannel // 2, affine=True))
            layers.append(nn.LeakyReLU(LR, inplace=True))
            self.mchannel = self.mchannel // 2

        # 256
        if self.g_ver == 0:
            layers.append(nn.ConvTranspose2d(self.mchannel, 3, kernel_size=4, stride=2, padding=1, bias=False))
        elif self.g_ver == 1:
            layers.append(nn.ConvTranspose2d(self.mchannel, 3, kernel_size=2, stride=2, bias=False))
        elif self.g_ver == 2:
            layers.append(nn.ConvTranspose2d(self.mchannel, 3, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False))

        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, base_channel=16, nlayers=4, LR=0.01):
        super(Discriminator, self).__init__()

        layers = list()

        layers.append(nn.Conv2d(3, base_channel, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(base_channel, affine=True))
        layers.append(nn.LeakyReLU(LR, inplace=True))

        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(base_channel * 2, affine=True))
            layers.append(nn.LeakyReLU(LR, inplace=True))
            base_channel = base_channel * 2

        layers.append(nn.Conv2d(base_channel, 1, kernel_size=3, stride=2, padding=1, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class CannyNet(nn.Module):
    def __init__(self, threshold=10.0, gpu=None):
        super(CannyNet, self).__init__()

        self.threshold = threshold
        self.gpu = gpu

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):

        img_r = img[:,0:1]
        img_g = img[:,1:2]
        img_b = img[:,2:3]

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        #blurred_img = torch.stack([blurred_img_r,blurred_img_g,blurred_img_b],dim=1)
        #blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag = grad_mag + torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag = grad_mag + torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation = grad_orientation + 180.0
        grad_orientation = torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        #pixel_range = torch.FloatTensor([range(pixel_count)])

        if self.gpu is not None:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)]).to(self.gpu)
        else:
            pixel_range = torch.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        #thin_edges[is_max==0] = 0.0
        thin_edges = thin_edges * (is_max == 1).type(torch.cuda.FloatTensor)
        #print(thin_edges)

        # THRESHOLD

        #thresholded = thin_edges.clone()
        #print((thin_edges < self.threshold))
        #thresholded[thin_edges < self.threshold] = 0.0
        thresholded = thin_edges * (thin_edges > self.threshold).type(torch.cuda.FloatTensor)
        #(thresholded.data[0, 0] > 0.0).type(torch.cuda.FloatTensor)

        #early_threshold = grad_mag.clone()
        #early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() # == early_threshold.size()

        # return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold
        return thresholded