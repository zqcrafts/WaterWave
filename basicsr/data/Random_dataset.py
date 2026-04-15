from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from os import path as osp
from basicsr.utils import img2tensor, scandir
import numpy as np
import torch
from einops import rearrange
from torch import nn as nn

class SampleGenerateGT(nn.Module):
    def __init__(self):
        super(SampleGenerateGT, self).__init__()
        # 进行构造函数中的初始化，例如加载图像数据
        self.data = None  # 用于存储图像数据

    def forward(self, xs, data):
        # data.shape [h, w, c] float
        with torch.no_grad():
            # self.data = torch.from_numpy(data).float()
            # print(data.shape)  # torch.Size([512, 512, 3])
            # print(xs.shape)  # torch.Size([262144, 2])

            self.data = data
            shape = data.shape
            xs = xs * torch.tensor([shape[1], shape[0]]).float()
            # print(torch.tensor([shape[1], shape[0]]).float().shape)  # size[2] # [w, h]
            # print(xs.shape)
            # exit()
            indices = xs.long()
            lerp_weights = xs - indices.float()

            x0 = indices[:, 0].clamp(min=0, max=shape[1]-1)
            y0 = indices[:, 1].clamp(min=0, max=shape[0]-1)
            x1 = (x0 + 1).clamp(max=shape[1]-1)
            y1 = (y0 + 1).clamp(max=shape[0]-1)
            
            return (
                self.data[y0, x0] * (1.0 - lerp_weights[:,0:1]) * (1.0 - lerp_weights[:,1:2]) +
                self.data[y0, x1] * lerp_weights[:,0:1] * (1.0 - lerp_weights[:,1:2]) +
                self.data[y1, x0] * (1.0 - lerp_weights[:,0:1]) * lerp_weights[:,1:2] +
                self.data[y1, x1] * lerp_weights[:,0:1] * lerp_weights[:,1:2]
            )



@DATASET_REGISTRY.register()
class RandomDataset(data.Dataset):

    def __init__(self, opt):
        super(RandomDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.flow_dir = opt['dataroot_flow']
        self.flow_conf_dir = opt['dataroot_flow_conf']
        self.filename_tmpl = '{}'
        self.paths = []
        

        # print(sorted(list(scandir(self.gt_folder))))
        # exit()
        for i in range(len(list(scandir(self.gt_folder)))):
            gt_path = sorted(list(scandir(self.gt_folder)))[i]
            gt_path = osp.join(self.gt_folder, gt_path)
            if i < len(list(scandir(self.gt_folder))) - 1:
                flow_path = sorted(list(scandir(self.flow_dir)))[i]
                flow_path = osp.join(self.flow_dir, flow_path)
                flow_conf_path = sorted(list(scandir(self.flow_conf_dir)))[i]
                flow_conf_path = osp.join(self.flow_conf_dir, flow_conf_path)
                self.paths.append(dict([('gt_path', gt_path), ('flow_path', flow_path), ('flow_conf_path', flow_conf_path)]))
            else:
                self.paths.append(dict([('gt_path', gt_path), ('flow_path', None), ('flow_conf_path', None)]))         



    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32. 

        # print(self.paths[index].keys())
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=False, float32=True)
        c, h, w = img_gt.shape
        img_gt = img_gt.view(3, -1).permute(1, 0)

        if self.opt['phase'] != 'train':
            pass

        # construct grid
        grid = np.indices((h, w)).astype(np.float32)
        grid[0,:,:] = grid[0,:,:] / h
        grid[1,:,:] = grid[1,:,:] / w
        # grid =  grid[[1, 0]]
        self.grid = torch.from_numpy(rearrange(grid, 'c h w -> (h w) c'))

        
        # get pseudo GT
        SampleGT= SampleGenerateGT()  
        # construct grid
        batch = torch.rand([512*512, 2], dtype=torch.float32)  # 随机产生位置
        img_gt = img_gt.view(1080, 1920, 3) 
        pseudo_gt = SampleGT(batch, img_gt)

        # construct temporal index
        frame_name = gt_path.split('/')[-1].split('.')[0]
        t_i = torch.tensor(float(frame_name) + 1.0, dtype=torch.float32)/len(self.paths)
        t_i = t_i.expand(pseudo_gt.shape[0]).unsqueeze(1)

        # flow
        if self.flow_dir:
            flow_path = self.paths[index]['flow_path']

            if flow_path is not None:
                flow=np.load(flow_path)  # (1, 2, h, w)
                flow =torch.from_numpy(flow).float()[:, [1, 0]]
                # flow =torch.from_numpy(flow).float()
                flow=flow.reshape(2,-1).transpose(1,0)

                flow[..., 0] = flow[..., 0] / h
                flow[..., 1] = flow[..., 1] / w
        
                flow_conf_path = self.paths[index]['flow_conf_path']
                flow_conf = np.load(flow_conf_path)
                flow_conf = torch.from_numpy(flow_conf).float()
                flow_conf = flow_conf.reshape(1,-1).transpose(1,0)
                flow_conf = flow_conf.sum(dim=-1) < 0.05
                flow[flow_conf] = 5
                flow = flow.view(1080, 1920, 2)  #  # torch.Size([262144, 2])
                pseudo_flow_gt = SampleGT(batch, flow)
            else:
                pseudo_flow_gt = torch.tensor(-1e5)


        batch = batch[:,[1,0]]
        # torch.set_printoptions(profile="full")
        # print(flow)   
        # exit()

        return {'gt': pseudo_gt, 'gt_path': gt_path, 'grid': batch, 't_i': t_i, 'flow': pseudo_flow_gt}


    def __len__(self):

        return len(self.paths)
