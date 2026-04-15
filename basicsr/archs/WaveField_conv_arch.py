import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample
import tinycudann as tcnn
import json
import matplotlib.pyplot as plt
import time

class Splitting1D(nn.Module):
    def __init__(self):
        super(Splitting1D, self).__init__()

        self.conv_even = lambda x: x[:, ::2]
        self.conv_odd = lambda x: x[:, 1::2]

    def forward(self, x):
        '''Returns the odd and even part'''

        return self.conv_even(x), self.conv_odd(x)


class WaveletHaar1D(nn.Module):
    def __init__(self, horizontal=True):
        super(WaveletHaar1D, self).__init__()
        self.split = Splitting1D()

    def forward(self, x):
        '''Returns the approximation and detail part'''
        (x_even, x_odd) = self.split(x)

        # Haar wavelet definition
        d = x_odd*0.5  - x_even*0.5
        c = x_odd*0.5  + x_even*0.5


        return (c, d)


class Splitting(nn.Module):
    def __init__(self, horizontal):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction
        self.horizontal = horizontal
        if(horizontal):
            self.conv_even = lambda x: x[:, :, :, ::2]
            self.conv_odd = lambda x: x[:, :, :, 1::2]
        else:
            self.conv_even = lambda x: x[:, :, ::2, :]
            self.conv_odd = lambda x: x[:, :, 1::2, :]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))


class WaveletHaar(nn.Module):
    def __init__(self, horizontal=True):
        super(WaveletHaar, self).__init__()
        self.split = Splitting(horizontal)

    def forward(self, x):
        '''Returns the approximation and detail part'''
        (x_even, x_odd) = self.split(x)

        # Haar wavelet definition
        d = x_odd*0.5  - x_even*0.5
        c = x_odd*0.5  + x_even*0.5
        return (c, d)


class WaveletHaar2D(nn.Module):
    def __init__(self):
        super(WaveletHaar2D, self).__init__()
        self.horizontal_haar = WaveletHaar(horizontal=True)
        self.vertical_haar = WaveletHaar(horizontal=False)

    def forward(self, x):
        '''Returns (LL, LH, HL, HH)'''
        (c, d) = self.horizontal_haar(x)
        (LL, LH) = self.vertical_haar(c)
        (HL, HH) = self.vertical_haar(d)
        return LL, (LH, HL, HH)


class ImplicitVideo_Hash(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=3,
                                     encoding_config=config["encoding"])
        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims + 3,
                                    n_output_dims=3,
                                    network_config=config["network_deform"])

    def forward(self, x):
        # print(self.encoder.n_output_dims)
        # exit()
        input = x
        input = self.encoder(input)
        input = torch.cat([x, input], dim=-1)
        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input)

        return x


class Deform_Hash3d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=3,
                                     encoding_config=config["encoding_deform3d"])
        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims + 3,
                                    n_output_dims=2,
                                    network_config=config["network_deform"])

    def forward(self, x, step=0, aneal_func=None):
        input = x
        input = self.encoder(input)
        if aneal_func is not None:
            input = torch.cat([x, aneal_func(input,step)], dim=-1)
        else:
            input = torch.cat([x, input], dim=-1)

        weight = torch.ones(input.shape[-1], device=input.device).cuda()
        x = self.decoder(weight * input) / 5

        return x


class AnnealedHash(nn.Module):
    def __init__(self, in_channels, annealed_step, annealed_begin_step=0, identity=True):

        super(AnnealedHash, self).__init__()
        self.N_freqs = 16
        self.in_channels = in_channels
        self.annealed = True
        self.annealed_step = annealed_step
        self.annealed_begin_step = annealed_begin_step
        self.index = torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        self.identity = identity
        self.index_2 = self.index.view(-1, 1).repeat(1, 2).view(-1)

    def forward(self, x_embed, step):

        if self.annealed_begin_step == 0:
            alpha = self.N_freqs * step / float(self.annealed_step)
        else:
            if step <= self.annealed_begin_step:
                alpha = 0
            else:
                alpha = self.N_freqs * (step - self.annealed_begin_step) / float(self.annealed_step)
        w = (1 - torch.cos(math.pi * torch.clamp(alpha * torch.ones_like(self.index_2) - self.index_2, 0, 1))) / 2
        out = x_embed * w.to(x_embed.device)

        return out


@ARCH_REGISTRY.register()
class WaveField_conv(nn.Module):
    def __init__(self, annealed_step=None, annealed_begin_step=None):
        super(WaveField_conv, self).__init__()

        self.embedding_hash = AnnealedHash(
                                in_channels=2,
                                annealed_step= annealed_step,
                                annealed_begin_step= annealed_begin_step)


        with open('basicsr/archs/hash.json') as f:
            config = json.load(f)
            self.Warp_estimation = Deform_Hash3d(config=config)
            self.VideoINR = ImplicitVideo_Hash(config=config)

        self.wave_s = WaveletHaar2D()
        self.wave_t = WaveletHaar1D()
        self.conv = nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        nn.init.constant_(self.conv.weight, 1.0)
        for param in self.conv.parameters():
            param.requires_grad = False

        self.step = 0

    def forward(self, grid, t_i, flow, num_frame):

        self.step += 1
        is_train = 0
        wave_order = 1

        if grid.shape[0] == 1:
            grid = grid.squeeze(0)
            t_i = t_i.squeeze(0)
            is_train = 1
            self.mk_t = torch.ones(flow.shape[0]).to('cuda')

        # start = time.time()
        xyt = torch.cat((grid, t_i), dim=-1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        output = self.VideoINR(xyt)
        # total_time = time.time() - start
        # print(total_time)

        if is_train:
            # 1-order wave: Considering 2 frames i.e. t and t+1
            if flow.max() > -1e2:  # temporal
                if wave_order > 0:
                    # flow_recify fw:flow_warp   r:recify
                    grid_fw_l1 = grid.clone() + flow.squeeze(0)
                    xyt_fw_l1 = torch.cat((grid_fw_l1, (t_i + float(1/num_frame))), dim=-1)
                    grid_r_l1 = self.Warp_estimation(xyt_fw_l1, step=self.step, aneal_func=self.embedding_hash) + grid_fw_l1
                    xyt_r_l1 = torch.cat((grid_r_l1, (t_i + float(1)/float(num_frame))), dim=-1)
                    # l1_0 (t+1)-th
                    mk_fw_l1 = torch.logical_and(self.mk_t, flow.squeeze(0).sum(dim=-1)< 3).unsqueeze(1)
                    output_l1 = self.VideoINR(xyt_r_l1)
                    # temporal
                    t_l1 = torch.stack((output.clone(), output_l1), dim=0).unsqueeze(0)
                    t_l1 = t_l1.permute(0,2,3,1).view(-1,2)
                    L_t, H_t = self.wave_t(t_l1)
                    L_t = L_t.view(1,1080,1920,3).permute(0,3,1,2)
                    H_t = H_t.view(1,1080,1920,3).permute(0,3,1,2)
            else:
                L_t = torch.zeros([1,3,1080,1920]).to('cuda')
                H_t = torch.zeros([1,3,1080,1920]).to('cuda')
                mk_fw_l1 = torch.zeros([1,1,1080,1920]).to('cuda')


            output = output.unsqueeze(0) # spatial
            L_s, (LH_s, HL_s, HH_s) = self.wave_s(output.view(1, 1080, 1920, 3).permute(0,3,1,2))
            H_s = LH_s + HL_s + HH_s

            # conv
            H_s_copy = self.conv(self.upsample(H_s.type(torch.cuda.FloatTensor)))
            mask_0 = torch.zeros(H_t.shape).to('cuda')
            mask_1 = torch.zeros(H_t.shape).to('cuda')
            th = 0.3
            mask_0[(H_s_copy < 0.1) & (H_s_copy > -0.1)] = 1
            mask_1[H_t > 0.0001] = 1
            mask_1[H_t < -0.0001] = 1
            mask = mask_1 + mask_0
            mask[mask < 2] = 0
            mask[mask!=0] = 1
            H_t = mask * H_t

            # return [L_s, H_s, L_t, H_t, output.view(1, 1080, 1920, 3).permute(0,3,1,2), mk_fw_l1.view(1,1080,1920,1).permute(0,3,1,2)]
            # print(mask)
            # print(mask.shape)
            return [L_s, H_s, L_t, H_t, output.view(1, 1080, 1920, 3).permute(0,3,1,2), mask]

        elif is_train==0:

            return output



