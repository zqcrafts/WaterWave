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

@DATASET_REGISTRY.register()
class CodeFDataset(data.Dataset):

    def __init__(self, opt):
        super(CodeFDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.flow_dir = opt['dataroot_flow']
        self.flow_conf_dir = opt['dataroot_flow_conf']
        self.filename_tmpl = '{}'
        self.paths = []

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
        grid =  grid[[1, 0]]
        self.grid = torch.from_numpy(rearrange(grid, 'c h w -> (h w) c'))

        # construct temporal index
        frame_name = gt_path.split('/')[-1].split('.')[0]
        t_i = torch.tensor(float(frame_name), dtype=torch.float32)/len(self.paths)
        t_i = t_i.expand(self.grid.shape[0]).unsqueeze(1)

        # flow
        if self.flow_dir:
            flow_path = self.paths[index]['flow_path']

            if flow_path is not None:
                flow=np.load(flow_path)  # (1, 2, h, w)
                # flow =torch.from_numpy(flow).float()[:, [1, 0]]
                flow =torch.from_numpy(flow).float()
                flow=flow.reshape(2,-1).transpose(1,0)
                # flow[..., 0] /= w*100
                # flow[..., 1] /= h*100
        
                flow_conf_path = self.paths[index]['flow_conf_path']
                flow_conf = np.load(flow_conf_path)
                flow_conf = torch.from_numpy(flow_conf).float()
                flow_conf = flow_conf.reshape(1,-1).transpose(1,0)
                flow_conf = flow_conf.sum(dim=-1) < 0.05
                flow[flow_conf] = 5

            else:
                flow = torch.tensor(-1e5)

        return {'gt': img_gt, 'gt_path': gt_path, 'grid': self.grid, 't_i': t_i, 'flow': flow}


    def __len__(self):

        return len(self.paths)
