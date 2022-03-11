import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config      as cfg


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, predict, label):
        l1loss = torch.mean(torch.abs(predict - label))
        return l1loss

class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, image, label):
        MSE = (image - label) * (image - label)
        if type(MSE) is np.ndarray:
            MSE = torch.from_numpy(MSE)
        MSE = torch.mean(MSE)
        PSNR = 10 * torch.log(1 / MSE) / torch.log(torch.Tensor([10.])).cuda()  # torch.log is log base e

        return PSNR


def loss_color(model, layers, device):       # Color Transform
    '''
    :param model:
    :param layers: layer name we want to use orthogonal regularization
    :param device: cpu or gpu
    :return: loss
    '''
    loss_orth = torch.tensor(0., dtype = torch.float32, device = device)
    params = {}
    for name, param in model.named_parameters():
        # k = '.'.join(name.split('.')[1:])
        # params[k] = param
        params[name] = param
    # for k,v in params.items():
    # print(k,v.shape)
    if cfg.ngpu > 1:
        ct = params['module.ct.w'].squeeze()
        cti = params['module.cti.w'].squeeze()
    else:
        ct = params['ct.w'].squeeze()
        cti = params['cti.w'].squeeze()
    weight_squared = torch.matmul(ct, cti)
    diag = torch.eye(weight_squared.shape[0], dtype=torch.float32, device=device)
    loss = ((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth

mse_criterion = torch.nn.MSELoss(reduction='mean')
l1_criterion = torch.nn.L1Loss()
def calc_Content_Loss(features, targets, weights=None, ltype='l2'):
    if ltype == 'l1':
        criterion = l1_criterion
    if ltype == 'l2':
        criterion = mse_criterion
    if weights is None:
        weights = [1 / len(features)] * len(features)
    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += criterion(f, t) * w
    return content_loss

def loss_wavelet(model, device): # Frequency Transform
    '''
    :param model:
    :param device: cpu or gpu
    :return: loss
    '''
    loss_orth = torch.tensor(0., dtype = torch.float32, device = device)
    params = {}
    for name, param in model.named_parameters():
        #k = '.'.join(name.split('.')[1:])
        #params[k] = param
        params[name] = param
    if cfg.ngpu > 1:
        ft = torch.cat([params['module.ft.w1'], params['module.ft.w2']], dim=0).squeeze()
        fti = torch.cat([params['module.fti.w1'],params['module.fti.w2']],dim= 0).squeeze()
    else:
        ft = torch.cat([params['ft.w1'], params['ft.w2']], dim=0).squeeze()
        fti = torch.cat([params['fti.w1'], params['fti.w2']], dim=0).squeeze()

    weight_squared = torch.matmul(ft, fti)
    diag = torch.eye(weight_squared.shape[1], dtype=torch.float32, device=device)
    loss=((weight_squared - diag) **2).sum()
    loss_orth += loss
    return loss_orth

def loss_ft_Conv(model, device):                            # Frequency Transform
    '''
    :param model:
    :param device: cpu or gpu
    :return: loss
    '''
    loss_orth = torch.tensor(0., dtype = torch.float32, device = device)
    params = {}
    for name, param in model.named_parameters():
        params[name] = param
    if cfg.ngpu > 1:
        ft = params['module.ft.net1.weight'].squeeze()
        fti = params['module.fti.net1.weight'].squeeze()
    else:
        ft = params['ft.net1.weight'].squeeze()
        fti = params['fti.net1.weight'].squeeze()
    weight_squared = torch.matmul(ft, fti)
    diag = torch.eye(weight_squared.shape[-1], dtype=torch.float32, device=device)
    loss=((weight_squared - diag) **2).mean() * 4 #.sum() #
    loss_orth += loss
    return loss_orth


# Define GAN loss: [gan | lsgan]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val  # 1.0
        self.fake_label_val = fake_label_val  # 0.0

        if self.gan_type == 'gan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if target_is_real:
            return torch.zeros_like(input).fill_(self.real_label_val)
        else:
            return torch.zeros_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


##########################
# Ohter Losses
##########################

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, input, target, mask):
        loss = self.criterion(input * mask, target * mask)
        return loss


class MultiscaleL1Loss(nn.Module):
    def __init__(self, scale=5):
        super(MultiscaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, input, target, mask=None):
        loss = 0
        for i in range(len(self.weights)):
            if mask is not None:
                loss += self.weights[i] * self.criterion(input * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights) - 1:
                input = self.downsample(input)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss

##########################
# Edge Losses
##########################
class EdgeLoss(nn.Module):
    def __init__(self, kernel_type='sobel', weight=0.1, c=4, criterion='L1'):
        super(EdgeLoss, self).__init__()
        self.weight = weight
        if kernel_type == 'sobel':
            sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
            sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
            sobel_x = sobel_x.reshape((1, 1, 3, 3))
            sobel_y = sobel_y.reshape((1, 1, 3, 3))
            sobel_x_kernel = np.tile(sobel_x, (c, c, 1, 1))
            sobel_y_kernel = np.tile(sobel_y, (c, c, 1, 1))
            sobel_x_kernel = torch.from_numpy(sobel_x_kernel)
            sobel_y_kernel = torch.from_numpy(sobel_y_kernel)
            self.conv_sobel_x = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_sobel_y = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv_sobel_x.weight = torch.nn.Parameter(sobel_x_kernel, requires_grad=False)
            self.conv_sobel_y.weight = torch.nn.Parameter(sobel_y_kernel, requires_grad=False)
        self.criterion = criterion

    def forward(self, image, label):
        img_x = self.conv_sobel_x(image)
        img_y = self.conv_sobel_y(image)
        label_x = self.conv_sobel_x(label)
        label_y = self.conv_sobel_y(label)
        img_gradient = torch.sqrt(torch.square(img_x)+torch.square(img_y))
        label_gradient = torch.sqrt(torch.square(label_x)+torch.square(label_y))
        if self.criterion == 'L1':
            l1loss = torch.mean(torch.abs(img_gradient - label_gradient))
            loss = l1loss * self.weight
        elif self.criterion == 'MSE':
            MSE = torch.square(img_gradient - label_gradient)
            if type(MSE) is np.ndarray:
                MSE = torch.from_numpy(MSE)
            MSE = torch.mean(MSE)
            loss = MSE * self.weight
        return loss # , img_gradient, label_gradient

def scale(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

import cv2
def main():
    edge = EdgeLoss(c=3)
    gt = np.transpose(cv2.imread('scene7_frame2_sRGB.png', -1),(2,0,1)).astype(np.float32) / 255.
    img = np.transpose(cv2.imread('scene7_frame2_sRGB_.png', -1),(2,0,1)).astype(np.float32) / 255.
    label = torch.from_numpy(np.expand_dims(gt, axis=0))
    predict = torch.from_numpy(np.expand_dims(img, axis=0))
    loss, img_gradient, label_gradient = edge(predict,label)
    print(loss)
    label = np.uint8(scale(label_gradient).numpy() * 255)[0]
    image = np.uint8(scale(img_gradient).numpy() * 255)[0]
    cv2.imwrite('scene7_frame2_sRGB_sobel.png',np.transpose(label,(1,2,0)))
    cv2.imwrite('scene7_frame2_sRGB_sobel_.png',np.transpose(image,(1,2,0)))

if __name__ == '__main__':
    main()