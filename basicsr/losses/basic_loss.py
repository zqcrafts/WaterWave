import torch
from torch import nn as nn
from torch.nn import functional as F
import cv2
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
import pywt


_reduction_modes = ['none', 'mean', 'sum']



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



@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

def rgb_to_gray(image):
    gray_image = (0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] +
                  0.114 * image[:, 2, :, :])
    gray_image = gray_image.unsqueeze(1)

    return gray_image


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, predict, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """

        return self.loss_weight * l1_loss(predict, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WaveL1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(WaveL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        [L_s, H_s, L_t, H_t, predict, _] = pred

        predict = predict.to(torch.float32)
        target = target.view(1, 1080, 1920, 3).permute(0,3,1,2)

        return self.loss_weight * l1_loss(predict, target, weight, reduction=self.reduction) * self.loss_weight


@LOSS_REGISTRY.register()
class GradDataLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradDataLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, predict, target, weight=None, **kwargs):

        predict = predict.to(torch.float32)

        pred = rgb_to_gray(predict)
        gt = rgb_to_gray(target)

        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device)

        gradient_a_x = torch.nn.functional.conv2d(pred.repeat(1,3,1,1), sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        gradient_a_y = torch.nn.functional.conv2d(pred.repeat(1,3,1,1), sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        # gradient_a_magnitude = torch.sqrt(gradient_a_x ** 2 + gradient_a_y ** 2)

        gradient_b_x = torch.nn.functional.conv2d(gt.repeat(1,3,1,1), sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        gradient_b_y = torch.nn.functional.conv2d(gt.repeat(1,3,1,1), sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        # gradient_b_magnitude = torch.sqrt(gradient_b_x ** 2 + gradient_b_y ** 2)

        pred_grad = torch.cat([gradient_a_x, gradient_a_y], dim=1)
        gt_grad = torch.cat([gradient_b_x, gradient_b_y], dim=1)

        # gradient_difference = torch.abs(pred_grad - gt_grad).mean(dim=1,keepdim=True)[mask].sum()/(mask.sum()+1e-8)
        gradient_difference = torch.abs(pred_grad - gt_grad).mean()

        return gradient_difference + l1_loss(predict, target, weight, reduction=self.reduction)



@LOSS_REGISTRY.register()
class GradLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(GradLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, predict, target, weight=None, **kwargs):

        [L_s, H_s, L_t, H_t, predict, _] = predict

        predict = predict.to(torch.float32)
        target = target.view(1, 1080, 1920, 3).permute(0,3,1,2)

        pred = rgb_to_gray(predict)
        gt = rgb_to_gray(target)

        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device)

        gradient_a_x = torch.nn.functional.conv2d(pred.repeat(1,3,1,1), sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        gradient_a_y = torch.nn.functional.conv2d(pred.repeat(1,3,1,1), sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        # gradient_a_magnitude = torch.sqrt(gradient_a_x ** 2 + gradient_a_y ** 2)

        gradient_b_x = torch.nn.functional.conv2d(gt.repeat(1,3,1,1), sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        gradient_b_y = torch.nn.functional.conv2d(gt.repeat(1,3,1,1), sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1), padding=1)/3
        # gradient_b_magnitude = torch.sqrt(gradient_b_x ** 2 + gradient_b_y ** 2)

        pred_grad = torch.cat([gradient_a_x, gradient_a_y], dim=1)
        gt_grad = torch.cat([gradient_b_x, gradient_b_y], dim=1)

        gradient_difference = torch.abs(pred_grad - gt_grad).mean()

        return gradient_difference *  self.loss_weight


@LOSS_REGISTRY.register()
class WaveLiftingLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(WaveLiftingLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.wave_t = WaveletHaar1D()
        self.wave_s = WaveletHaar2D()

    def forward(self, predict, target, weight=None, **kwargs):

        # predict: (t c h w)
        # target: (t c h w)
        [L_s, H_s, L_t, H_t, _] = predict
        target = target.view(-1, 1080, 1920, 3).permute(0,3,1,2)

        b, c, h, w = target.shape
        L_gt, (LH_gt, HL_gt, HH_gt) = self.wave_s(target)
        H_gt = LH_gt + HL_gt + HH_gt

        # print(pLLL.shape)
        # cv2.imwrite('1.png', tLLL.permute(1,2,0).detach().cpu().numpy()*255)
        # exit()

        wave_loss = l1_loss(L_s, L_gt, torch.ones(L_gt.shape).to('cuda')*1.0, reduction=self.reduction)
        # + l1_loss(H_t, torch.zeros(H_t.shape).to('cuda'), torch.ones(H_t.shape).to('cuda')*50.0, reduction=self.reduction)
        + l1_loss(H_s, H_gt, torch.ones(H_gt.shape).to('cuda')*1.0, reduction=self.reduction)
        # + l1_loss(H_t, torch.zeros(H_t.shape).to('cuda'), torch.ones(H_t.shape).to('cuda')*50.0, reduction=self.reduction)
        # + l1_loss(pLHH, 0, weight, reduction=self.reduction)

        # wave_loss = wave_loss

        # print(wave_loss)
        # print(wave_loss.shape)
        # exit()

        return wave_loss


@LOSS_REGISTRY.register()
class WaveLoss_mix_mask(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(WaveLoss_mix_mask, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.wave_t = WaveletHaar1D()
        self.wave_s = WaveletHaar2D()
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)

    def forward(self, predict, target, weight=None, **kwargs):

        # predict: (t c h w)
        # target: (t c h w)
        [L_s, H_s, L_t, H_t, _, mask] = predict
        target = target.view(-1, 1080, 1920, 3).permute(0,3,1,2)

        b, c, h, w = target.shape
        L_gt, (LH_gt, HL_gt, HH_gt) = self.wave_s(target)
        H_gt = LH_gt + HL_gt + HH_gt

        # cv2.imwrite('1.png', tLLL.permute(1,2,0).detach().cpu().numpy()*255)
        # exit()
        wave_loss = l1_loss(L_s, L_gt, reduction=self.reduction) * self.loss_weight
        # l1_loss(H_t, torch.zeros(H_t.shape).to('cuda'), reduction=self.reduction)*5.0
        # + l1_loss(H_s, H_gt, reduction=self.reduction) * self.loss_weight
        # + l1_loss(L_s*(self.downsample(1-mask)), L_gt*(self.downsample(1-mask)), reduction=self.reduction) * self.loss_weight
        # + l1_loss(L_s, L_gt, reduction=self.reduction) * self.loss_weight
        # + l1_loss(L_t, target, reduction=self.reduction) * self.loss_weight
        # + l1_loss(H_t*fw_mk, torch.zeros(H_t.shape).to('cuda')*fw_mk, torch.ones(H_t.shape).to('cuda')*10.0, reduction=self.reduction)

        return wave_loss

@LOSS_REGISTRY.register()
class WaveLoss_Lt(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(WaveLoss_Lt, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.wave_t = WaveletHaar1D()
        self.wave_s = WaveletHaar2D()

    def forward(self, predict, target, weight=None, **kwargs):

        # predict: (t c h w)
        # target: (t c h w)
        [L_s, H_s, L_t, H_t, _, fw_mk] = predict
        target = target.view(-1, 1080, 1920, 3).permute(0,3,1,2)

        b, c, h, w = target.shape
        L_gt, (LH_gt, HL_gt, HH_gt) = self.wave_s(target)
        H_gt = LH_gt + HL_gt + HH_gt

        # cv2.imwrite('1.png', tLLL.permute(1,2,0).detach().cpu().numpy()*255)
        # exit()

        wave_loss = l1_loss(H_t, torch.zeros(H_t.shape).to('cuda'), reduction=self.reduction)*5.0
        + l1_loss(L_t, target, reduction=self.reduction)*self.loss_weight

        # + l1_loss(H_t*fw_mk, torch.zeros(H_t.shape).to('cuda')*fw_mk, torch.ones(H_t.shape).to('cuda')*10.0, reduction=self.reduction)

        return wave_loss

@LOSS_REGISTRY.register()
class regular(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(regular, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, predict, target, weight=None, **kwargs):

        # predict: (t c h w)
        # target: (t c h w)
        [L_s, H_s, L_t, H_t, _, fw_mk] = predict
        wave_loss = l1_loss(H_t, torch.zeros(H_t.shape).to('cuda'), reduction=self.reduction)*self.loss_weight

        return wave_loss


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
