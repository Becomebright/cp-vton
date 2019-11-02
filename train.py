# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import VGGLoss, load_checkpoint, save_checkpoint, WUTON, Discriminator

from tensorboardX import SummaryWriter
from visualization import board_add_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)

    parser.add_argument("--dataroot", default="data")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=100)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--n_iter", type=int, default=100000, help='Number of iteration cycles')
    parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')
    parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
    parser.add_argument('--cuda', type='bool', default=True, help='enables cuda')
    parser.add_argument('--penalty', type=float, default=10, help='Gradient penalty parameter for WGAN-GP')

    opt = parser.parse_args()
    return opt


def train(opt, train_loader, G, D, board):
    G.train()
    D.train()

    # Criterion
    criterionWarp = nn.L1Loss()
    criterionPerceptual = VGGLoss()
    criterionL1 = nn.L1Loss()
    BCE_stable = nn.BCEWithLogitsLoss()

    # Everything cuda
    if opt.cuda:
        G = G.cuda()
        D = D.cuda()
        criterionWarp = criterionWarp.cuda()
        criterionPerceptual = criterionPerceptual.cuda()
        criterionL1 = criterionL1.cuda()
        BCE_stable.cuda()

    # Optimizers
    optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Fitting model
    for i in range(opt.n_iter):
        iter_start_time = time.time()

        ########################
        # (1) Update D network #
        ########################

        for p in D.parameters():
            p.require_grad = True

        for t in range(opt.Diters):
            D.zero_grad()

            inputs = train_loader.next_batch()
            pa = inputs['image'].cuda()  # reference image
            ap = inputs['agnostic'].cuda()
            cb = inputs['another_cloth'].cuda()  # TODO: another cloth

            current_batch_size = pa.size(0)
            ya_pred = D(pa)
            _, pb_fake = G(cb, ap)
            yb_pred_fake = D(pb_fake.detach())  # Detach y_pred_fake from the neural network G and put it inside D
            ya = torch.ones(current_batch_size)
            yb = torch.zeros(current_batch_size)

            errD = (BCE_stable(ya_pred - torch.mean(yb_pred_fake), ya) +
                    BCE_stable(yb_pred_fake - torch.mean(ya_pred), yb)) / 2
            errD.backward()

            # Gradient penalty
            u = torch.FloatTensor(current_batch_size, 1, 1, 1).uniform_(0, 1)
            x_both = pa.data * u + pb_fake.data * (1. - u)
            if opt.cuda:
                x_both = x_both.cuda()
            # We only want the gradients with respect to x_both
            x_both = Variable(x_both, requires_grad=True)
            grad = torch.autograd.grad(outputs=D(x_both), inputs=x_both,
                                       grad_outputs=torch.ones(current_batch_size), retain_graph=True,
                                       create_graph=True, only_inputs=True)[0]
            # We need to norm 3 times (over n_colors x image_size x image_size) to get only a vector of size
            # "batch_size"
            grad_penalty = opt.penalty * ((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
            grad_penalty.backward()

            optimizerD.step()

        ########################
        # (2) Update G network #
        ########################

        for p in D.parameters():
            p.requires_grad = False

        for step in range(opt.Giters):
            G.zero_grad()

            inputs = train_loader.next_batch()
            pa = inputs['image'].cuda()  # reference image
            # pose_a = inputs['pose_image'].cuda()
            ap = inputs['agnostic'].cuda()
            ca = inputs['cloth'].cuda()
            cb = inputs['another_cloth'].cuda()  # TODO: another cloth
            parse_cloth = inputs['parse_cloth'].cuda()  # c_{a,p}
            parse_neck = inputs['parse_neck'].cuda()

            current_batch_size = pa.size(0)

            grid_a, pa_fake = G(ap, ca)
            warped_cloth_a = F.grid_sample(ca, grid_a, padding_mode='border')
            grid_b, pb_fake = G(ap, cb)
            warped_cloth_b = F.grid_sample(cb, grid_b, padding_mode='border')

            ya_pred = D(pa)
            yb_pred_fake = D(pb_fake)
            ya = torch.ones(current_batch_size)
            yb = torch.zeros(current_batch_size)

            visuals = [     # TODO
                [cb, warped_cloth_b, pb_fake],
                [ca, warped_cloth_a, pa_fake],
                [parse_neck, ap, pa]
                # [cb, (warped_cloth_b + pa) * 0.5, pa],
                # [ca, warped_cloth_a, parse_cloth],
                # [parse_neck, (warped_cloth_a + pa) * 0.5, pa]
            ]

            # Non-saturating
            l_adv = (BCE_stable(ya_pred - torch.mean(yb_pred_fake), yb) +
                    BCE_stable(yb_pred_fake - torch.mean(ya_pred), ya)) / 2
            l_warp = criterionWarp(warped_cloth_a, parse_cloth)
            l_perceptual = criterionPerceptual(pa_fake, pa)
            l_L1 = criterionL1(pa_fake, pa)
            loss = l_warp + l_perceptual + l_L1 + l_adv

            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()

            if (step + 1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step + 1)
                board.add_scalar('metric', loss.item(), step + 1)
                t = time.time() - iter_start_time
                print('step: %8d, time: %.3f, loss: %4f, l_warp: %.4f, l_vgg: %.4f, l_L1: %.4f, l_adv: %.4f'
                      % (step + 1, t, loss.item(), l_warp.item(), l_perceptual.item(), l_L1.item(), l_adv.item()),
                      flush=True)

            if (step + 1) % opt.save_count == 0:
                save_checkpoint(G, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


# def train_gmm(opt, train_loader, cgm, unet, board):
#     cgm.cuda()
#     cgm.train()
#
#     # criterion
#     criterionL1 = nn.L1Loss()
#
#     # optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#
#     for step in range(opt.keep_step + opt.decay_step):
#         iter_start_time = time.time()
#         inputs = train_loader.next_batch()
#
#         im = inputs['image'].cuda()
#         im_pose = inputs['pose_image'].cuda()
#         im_h = inputs['head'].cuda()
#         shape = inputs['shape'].cuda()
#         agnostic = inputs['agnostic'].cuda()
#         c = inputs['cloth'].cuda()
#         # cm = inputs['cloth_mask'].cuda()
#         im_c = inputs['parse_cloth'].cuda()
#         im_g = inputs['grid_image'].cuda()
#         parse_neck = inputs['parse_neck'].cuda()
#
#         grid, theta = model(agnostic, c)
#         warped_cloth = F.grid_sample(c, grid, padding_mode='border')
#         # warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros')
#
#         visuals = [
#             [im_h, shape, im_pose],
#             [c, warped_cloth, im_c],
#             # [agnostic, (warped_cloth + im) * 0.5, im]
#             [agnostic, (warped_cloth + im) * 0.5, im]
#         ]
#
#         loss = criterionL1(warped_cloth, im_c)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (step + 1) % opt.display_count == 0:
#             board_add_images(board, 'combine', visuals, step + 1)
#             board.add_scalar('metric', loss.item(), step + 1)
#             t = time.time() - iter_start_time
#             print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss.item()), flush=True)
#
#         # if (step + 1) % opt.save_count == 0:
#         #     save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))
#
#
# def train_tom(opt, train_loader, model, board):
#     model.cuda()
#     model.train()
#
#     # criterion
#     criterionL1 = nn.L1Loss()
#     criterionVGG = VGGLoss()
#     criterionMask = nn.L1Loss()
#
#     # optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0 -
#                                                                                     max(0,
#                                                                                         step - opt.keep_step) / float(
#         opt.decay_step + 1))
#
#     for step in range(opt.keep_step + opt.decay_step):
#         iter_start_time = time.time()
#         inputs = train_loader.next_batch()
#
#         im = inputs['image'].cuda()
#         im_pose = inputs['pose_image']
#         im_h = inputs['head']
#         shape = inputs['shape']
#
#         agnostic = inputs['agnostic'].cuda()
#         c = inputs['cloth'].cuda()
#         cm = inputs['cloth_mask'].cuda()
#
#         outputs = model(torch.cat([agnostic, c], 1))
#         p_rendered, m_composite = torch.split(outputs, 3, 1)
#         p_rendered = F.tanh(p_rendered)
#         m_composite = F.sigmoid(m_composite)
#         p_tryon = c * m_composite + p_rendered * (1 - m_composite)
#
#         visuals = [[im_h, shape, im_pose],
#                    [c, cm * 2 - 1, m_composite * 2 - 1],
#                    [p_rendered, p_tryon, im]]
#
#         loss_l1 = criterionL1(p_tryon, im)
#         loss_vgg = criterionVGG(p_tryon, im)
#         loss_mask = criterionMask(m_composite, cm)
#         loss = loss_l1 + loss_vgg + loss_mask
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (step + 1) % opt.display_count == 0:
#             board_add_images(board, 'combine', visuals, step + 1)
#             board.add_scalar('metric', loss.item(), step + 1)
#             board.add_scalar('L1', loss_l1.item(), step + 1)
#             board.add_scalar('VGG', loss_vgg.item(), step + 1)
#             board.add_scalar('MaskL1', loss_mask.item(), step + 1)
#             t = time.time() - iter_start_time
#             print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
#                   % (step + 1, t, loss.item(), loss_l1.item(),
#                      loss_vgg.item(), loss_mask.item()), flush=True)
#
#         if (step + 1) % opt.save_count == 0:
#             save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


def main():
    opt = get_opt()
    print(opt)
    print("Start to train stage: %s, named: %s!" % (opt.stage, opt.name))

    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.name))

    # create model & train & save the final checkpoint
    G = WUTON(opt)
    D = Discriminator()
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):  # TODO
        load_checkpoint(G, opt.checkpoint)
    train(opt, train_loader, G, D, board)
    save_checkpoint(G, os.path.join(opt.checkpoint_dir, opt.name, 'wuton_final.pth'))

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
