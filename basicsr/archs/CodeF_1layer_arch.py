import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np
from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample
import tinycudann as tcnn
import json


class ImplicitVideo_Hash(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = tcnn.Encoding(n_input_dims=3,
                                     encoding_config=config["encoding"])
        self.decoder = tcnn.Network(n_input_dims=self.encoder.n_output_dims + 3,
                                    n_output_dims=3,
                                    network_config=config["network_deform"])

    def forward(self, x):
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
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
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
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """

        if self.annealed_begin_step == 0:
            # calculate the w for each freq bands
            alpha = self.N_freqs * step / float(self.annealed_step)
        else:
            if step <= self.annealed_begin_step:
                alpha = 0
            else:
                alpha = self.N_freqs * (step - self.annealed_begin_step) / float(self.annealed_step)

        w = (1 - torch.cos(math.pi * torch.clamp(alpha * torch.ones_like(self.index_2) - self.index_2, 0, 1))) / 2
        
        # print(x_embed.shape)  # torch.Size([2073600, 32])
        out = x_embed * w.to(x_embed.device)
        # exit()
        return out



@ARCH_REGISTRY.register()
class CodeF_1layer(nn.Module):
    def __init__(self, annealed_step=None, annealed_begin_step=None):
        super(CodeF_1layer, self).__init__()

        self.embedding_hash = AnnealedHash(
                                in_channels=2,
                                annealed_step= annealed_step,
                                annealed_begin_step= annealed_begin_step)
        

        with open('basicsr/archs/hash.json') as f:
            config = json.load(f)
            self.Warp_estimation = Deform_Hash3d(config=config)
            self.VideoINR = ImplicitVideo_Hash(config=config)

        self.step = 0

        # self.mk_t = torch.ones((1080*1920)).to('cuda')

            


    def forward(self, grid, t_i, flow):

        self.step += 1
        is_train = 0

        if grid.shape[0] == 1:
            grid = grid.squeeze(0)
            t_i = t_i.squeeze(0)
            is_train = 1
            self.mk_t = torch.ones(flow.shape[0]).to('cuda')
            
        xyt = torch.cat((grid, t_i), dim=-1)

        output = self.VideoINR(xyt)
        # deform = self.Warp_estimation(xyt, step=self.step, aneal_func=self.embedding_hash)
        # deformed_grid = grid.squeeze() + deform

        flow_loss = torch.zeros((2, 100, 3)).to('cuda')
        
        if is_train and flow.max() > -1e2:

            grid_2 = grid.clone() + flow.squeeze(0)
            xyt_2 = torch.cat((grid_2, (t_i + float(1)/float(48))), dim=-1)
            deformed_grid_new = self.Warp_estimation(xyt_2, step=self.step, aneal_func=self.embedding_hash) + grid_2
            xyt_3 = torch.cat((deformed_grid_new, (t_i + float(1)/float(48))), dim=-1)

            mk_flow_t = torch.logical_and(self.mk_t, flow.squeeze(0).sum(dim=-1)< 3).unsqueeze(1)
            torch.set_printoptions(profile="full")
            output_2 = self.VideoINR(xyt_3)
            flow_loss = (output.clone().unsqueeze(0) * mk_flow_t, output_2.unsqueeze(0) * mk_flow_t)
             

        if is_train:
            output = output.unsqueeze(0)


        return output, flow_loss

