# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
import os
import numpy as np
from collections import OrderedDict

from human_parsing.model import network


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=5, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FeatureExtraction, self).__init__()

        self.layers = nn.ModuleList()
        in_channel = input_nc
        out_channel = ngf
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                norm_layer(out_channel),
                nn.Conv2d(out_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.ReLU(True),
                norm_layer(out_channel)
            ))
            in_channel = out_channel
            out_channel *= 2
            init_weights(self.layers[i], init_type='normal')

    def forward(self, x):
        self.features = []
        for layer in self.layers:
            x = layer(x)
            self.features.append(x)
        self.features.pop()
        return x


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 2 * 1, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()
            self.tanh.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x


class AffineGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size)


class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        # if use_cuda:
        #     self.grid_X = self.grid_X.cuda()
        #     self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            # if use_cuda:
            #     self.P_X = self.P_X.cuda()
            #     self.P_Y = self.P_Y.cuda()
            #     self.P_X_base = self.P_X_base.cuda()
            #     self.P_Y_base = self.P_Y_base.cuda()

    def forward(self, theta):
        grid_X = self.grid_X
        grid_Y = self.grid_Y
        # cuda
        if self.use_cuda:
            grid_X = grid_X.cuda()
            grid_Y = grid_Y.cuda()
        warped_grid = self.apply_transformation(theta, torch.cat((grid_X, grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1  # make diagonal 1 to avoid NaN in log computation
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        # if self.use_cuda:
        #     Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        P_X_base = self.P_X_base
        P_Y_base = self.P_Y_base
        P_X = self.P_X
        P_Y = self.P_Y
        Li = self.Li
        # cuda
        if self.use_cuda:
            P_X_base = P_X_base.cuda()
            P_Y_base = P_Y_base.cuda()
            P_X = P_X.cuda()
            P_Y = P_Y.cuda()
            Li = Li.cuda()

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        # print('Q_X: ' + str(Q_X.device))
        # print('P_X_base: ' + str(self.P_X_base.device))
        Q_X = Q_X + P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(Li[:, :self.N, :self.N].expand((batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(Li[:, self.N:, :self.N].expand((batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(
            points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
# model = UnetGenerator(3, 4, num_downs=6, ngf=64, norm_layer=nn.InstanceNorm2d)
class UnetGenerator(nn.Module):
    def __init__(self, opt, input_nc=3, output_nc=3, num_downs=5, ngf=16,
                 norm_layer=nn.BatchNorm2d):
        super(UnetGenerator, self).__init__()

        self.use_cuda = opt.cuda
        self.extractionA = FeatureExtraction(input_nc, ngf, num_downs, nn.InstanceNorm2d)
        self.extractionB = FeatureExtraction(input_nc, ngf, num_downs, nn.InstanceNorm2d)
        self.decoder = Decoder(input_nc=ngf * (2 ** num_downs), output_nc=output_nc,
                               num_ups=num_downs, norm_layer=norm_layer)
        # self.l2norm = FeatureL2Norm()

    def forward(self, inputA, inputB, theta):
        outputA = self.extractionA(inputA)
        outputB = self.extractionB(inputB)
        for i, f in enumerate(self.extractionA.features):
            gridGen = TpsGridGen(f.size(2), f.size(3), use_cuda=self.use_cuda, grid_size=5)
            grid = gridGen(theta)
            self.extractionA.features[i] = F.grid_sample(f, grid, padding_mode='zeros')
        return self.decoder(outputA, self.extractionA.features, outputB, self.extractionB.features)


class Decoder(nn.Module):
    def __init__(self, input_nc=512, output_nc=3, num_ups=5, norm_layer=nn.InstanceNorm2d):
        super(Decoder, self).__init__()

        use_bias = norm_layer == nn.InstanceNorm2d
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Conv2d(input_nc, input_nc // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(input_nc // 2),
            nn.ConvTranspose2d(input_nc // 2, input_nc // 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.ReLU(True),
            norm_layer(input_nc // 2)
        ))

        in_channel = input_nc
        for i in range(num_ups - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channel, in_channel // 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.ReLU(True),
                norm_layer(in_channel // 2),
                nn.ConvTranspose2d(in_channel // 2, in_channel // 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
                nn.ReLU(True),
                norm_layer(in_channel // 4)
            ))
            in_channel //= 2

        self.conv11 = nn.Conv2d(in_channel // 2, output_nc, kernel_size=1, stride=1, padding=0)

    def forward(self, xA, featuresA, xB, featuresB):
        x = torch.cat([xA, xB], 1)
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = self.layers[0](x)
            else:
                x = torch.cat([featuresA[-i], featuresB[-i], x], 1)
                x = self.layers[i](x)
        x = self.conv11(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):
    # TODO: PatchGAN??
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, kernel_size=(8, 6), stride=1, padding=0),
        )
        init_weights(self.model, init_type='normal')

    def forward(self, x):
        return self.model(x).view(-1)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class CGM(nn.Module):
    """ Convoluional Geometric Matcher
    """

    def __init__(self, opt):
        super(CGM, self).__init__()
        self.extractionA = FeatureExtraction(3, ngf=16, n_layers=5, norm_layer=nn.BatchNorm2d)
        self.extractionB = FeatureExtraction(3, ngf=16, n_layers=5, norm_layer=nn.BatchNorm2d)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=48, output_dim=2 * opt.grid_size ** 2, use_cuda=opt.cuda)
        # self.gridGen = TpsGridGen(opt.fine_height, opt.fine_width, use_cuda=True, grid_size=opt.grid_size)

    def forward(self, cloth, masked_person):
        featureA = self.extractionA(cloth)
        featureB = self.extractionB(masked_person)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA, featureB)
        theta = self.regression(correlation)

        return theta


class HumanParser(nn.Module):
    def __init__(self, opt, num_classes=20,
                 restore_weight='human_parsing/models/exp-schp-201908261155-lip.pth'):
        super(HumanParser, self).__init__()
        self.model = network(num_classes=num_classes, pretrained=None)
        state_dict = torch.load(restore_weight)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
        self.opt = opt
        self.model.load_state_dict(new_state_dict)
        self.mean = torch.FloatTensor([0.406, 0.456, 0.485]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.FloatTensor([0.225, 0.224, 0.229]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.upsample = torch.nn.Upsample(size=[473, 473], mode='bilinear', align_corners=True)  # to (473, 473)
        self.downsample = torch.nn.Upsample(size=[opt.fine_height, opt.fine_width], mode='bilinear', align_corners=True)  # to (256, 192)

    def _normalize(self, x):
        """
        :param x: (N,C,H,W)
        :return: (N,C,H,W)
        """
        mean = self.mean
        std = self.std
        if self.opt.cuda:
            mean = mean.cuda()
            std = std.cuda()
        return (x - mean) / std

    def forward(self, image):
        """
        :param image: (N, C, H, W); [-1, 1]
        :return: parsed_image: (N, num_classes, H, W)
        """
        img = image.clone()
        img = (img + 1.) / 2.  # [-1,1] -> [0,1]
        img = self._normalize(img)
        img = self.upsample(img)
        parse = self.model(img)  # (N,C,H,W)
        parse = self.downsample(parse)
        parse = parse.argmax(1)  # (N,H,W)
        del img
        return parse


class WUTON(nn.Module):
    def __init__(self, opt):
        super(WUTON, self).__init__()
        self.cgm = CGM(opt)
        self.gridGen = TpsGridGen(opt.fine_height, opt.fine_width, use_cuda=opt.cuda, grid_size=opt.grid_size)
        self.unet = UnetGenerator(opt=opt)  # [-1, 1]

    def forward(self, cloth, masked_person):
        theta = self.cgm(cloth, masked_person)
        grid = self.gridGen(theta)
        warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
        fake_person = self.unet(cloth, masked_person, theta)

        return warped_cloth, fake_person


def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model.load_state_dict(torch.load(checkpoint_path))
    model.cuda()
