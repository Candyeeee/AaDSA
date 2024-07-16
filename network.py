import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import Variable
import numpy as np
from torchvision import models


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_weights(net, init_type='normal', gain=0.02):
    from torch.nn import init
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                             n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG




##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=6,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='zero'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf #* (2 ** n_local_enhancers)
        self.model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer)

        ###### local enhancer layers #####
        ngf_global = ngf // 2  # * (2 ** (n_local_enhancers - n))
        model_downsample = [nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=3),
                            norm_layer(ngf_global), nn.ReLU(True),
                            nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                            norm_layer(ngf_global * 2), nn.ReLU(True)]
        self.model_downsample = nn.Sequential(*model_downsample)

        model_local = []
        ### residual blocks
        for i in range(n_blocks_local):
            model_local += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        ### upsample
        # model_local += [
        #     nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
        #     norm_layer(ngf_global), nn.ReLU(True)]
        model_local += [nn.UpsamplingNearest2d(scale_factor=2),
                        nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=3, stride=1, padding=1),
                        norm_layer(ngf_global), nn.ReLU(True)]

        ### final convolution
        model_local += [nn.Conv2d(ngf_global, output_nc, kernel_size=7, padding=3)]

        self.model_local = nn.Sequential(*model_local)

    def forward(self, input, input_downsampled=None):
        ### create input pyramid
        if input_downsampled is None:
            input_downsampled = F.interpolate(input, scale_factor=0.5, mode='bilinear')

        out1, feat1 = self.model_global(input_downsampled)

        feat0 = self.model_downsample(input)
        feat0 = feat0 + feat1
        out0 = self.model_local(feat0)

        return out0, out1


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='zero', upsample_type='nearest', skip_connection=True):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()

        self.skip_connection = skip_connection
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        conv0 = [nn.Conv2d(input_nc, ngf, kernel_size=5, padding=2, bias=False), norm_layer(ngf), activation]
        self.conv0 = nn.Sequential(*conv0)

        ### downsample
        mult = 1
        conv_down1 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=5, stride=2, padding=2, bias=False),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down1 = nn.Sequential(*conv_down1)

        mult = 2
        conv_down2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=5, stride=2, padding=2, bias=False),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down2 = nn.Sequential(*conv_down2)

        mult = 4
        conv_down3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                      norm_layer(ngf * mult * 2), activation]
        self.conv_down3 = nn.Sequential(*conv_down3)

        mult = 8
        ### resnet blocks
        self.resnetBlock1 = ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        self.resnetBlock2 = SmoothDilatedResidualBlock(ngf * mult, dilation=2, activation=activation, norm_layer=norm_layer)
        self.resnetBlock3 = SmoothDilatedResidualBlock(ngf * mult, dilation=2, activation=activation, norm_layer=norm_layer)
        self.resnetBlock4 = SmoothDilatedResidualBlock(ngf * mult, dilation=4, activation=activation, norm_layer=norm_layer)
        self.resnetBlock5 = SmoothDilatedResidualBlock(ngf * mult, dilation=4, activation=activation, norm_layer=norm_layer)

        self.resnetBlock6 = ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)

        ### upsample
        if upsample_type=='convt':
            convt_up3 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1,
                                            output_padding=0, bias=False),
                         norm_layer(int(ngf * mult / 2)), activation]
            self.convt_up3 = nn.Sequential(*convt_up3)
        else:
            convt_up3 = [nn.UpsamplingNearest2d(scale_factor=2),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=False),
                         norm_layer(int(ngf * mult / 2)), activation]
            self.convt_up3 = nn.Sequential(*convt_up3)

        mult = 4
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv3 = [nn.Conv2d(in_channels, ngf * mult, kernel_size=3, stride=1, padding=1, bias=False),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv3 = nn.Sequential(*decoder_conv3)

        if upsample_type == 'convt':
            convt_up2 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1,
                                            output_padding=0, bias=False),
                         norm_layer(int(ngf * mult / 2)), activation]
            self.convt_up2 = nn.Sequential(*convt_up2)
        else:
            convt_up2 = [nn.UpsamplingNearest2d(scale_factor=2),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=False),
                         norm_layer(int(ngf * mult / 2)), activation]
            self.convt_up2 = nn.Sequential(*convt_up2)

        mult = 2
        if skip_connection:
            in_channels = ngf * mult * 2
        else:
            in_channels = ngf * mult
        decoder_conv2 = [nn.Conv2d(in_channels, ngf * mult, kernel_size=5, stride=1, padding=2, bias=False),
                         norm_layer(ngf * mult), activation]
        self.decoder_conv2 = nn.Sequential(*decoder_conv2)

        if upsample_type == 'convt':
            convt_up1 = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4, stride=2, padding=1,
                                            output_padding=0, bias=False),
                         norm_layer(int(ngf * mult / 2)), activation]
            self.convt_up1 = nn.Sequential(*convt_up1)
        else:
            convt_up1 = [nn.UpsamplingNearest2d(scale_factor=2),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=False),
                         norm_layer(int(ngf * mult / 2)), activation]
            self.convt_up1 = nn.Sequential(*convt_up1)

        if skip_connection:
            in_channels = ngf * 2
        else:
            in_channels = ngf
        self.decoder_conv1 = nn.Conv2d(in_channels, output_nc, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv_down1(x0)
        x2 = self.conv_down2(x1)
        x3 = self.conv_down3(x2)

        x3 = self.resnetBlock1(x3)
        x3 = self.resnetBlock2(x3)
        x3 = self.resnetBlock3(x3)
        x3 = self.resnetBlock4(x3)
        x3 = self.resnetBlock5(x3)
        x3 = self.resnetBlock6(x3)

        x4 = self.convt_up3(x3)
        if self.skip_connection:
            x4 = torch.cat((x4, x2), dim=1)
        x4 = self.decoder_conv3(x4)

        x5 = self.convt_up2(x4)
        if self.skip_connection:
            x5 = torch.cat((x5, x1), dim=1)
        x5 = self.decoder_conv2(x5)

        x6 = self.convt_up1(x5)
        if self.skip_connection:
            x6 = torch.cat((x6, x0), dim=1)
        out = self.decoder_conv1(x6)

        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class SmoothDilatedResidualBlock(nn.Module):
    def __init__(self, dim, dilation, norm_layer, activation=nn.ReLU(True)):
        super(SmoothDilatedResidualBlock, self).__init__()

        conv_block = [ShareSepConv(dilation * 2 - 1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                      norm_layer(dim, affine=True),
                      activation,
                      ShareSepConv(dilation * 2 - 1),
                      nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
                      norm_layer(dim, affine=True)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b + 1] == int(i)).nonzero()  # n x 4
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean



