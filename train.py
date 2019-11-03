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

# from apex import amp


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
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
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

    pa = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    ap = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    ca = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    cb = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    parse_cloth = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    parse_neck = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    ya = torch.FloatTensor(opt.batch_size)
    yb = torch.FloatTensor(opt.batch_size)
    pa_fake = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    pb_fake = torch.FloatTensor(opt.batch_size, 3, opt.fine_height, opt.fine_width)
    u = torch.FloatTensor(opt.batch_size, 1, 1, 1)
    grad_outputs = torch.ones(opt.batch_size)

    # Everything cuda
    G.cuda()
    D.cuda()
    criterionWarp = criterionWarp.cuda()
    criterionPerceptual = criterionPerceptual.cuda()
    criterionL1 = criterionL1.cuda()
    BCE_stable.cuda()
    pa = pa.cuda()
    ap = ap.cuda()
    ca = ca.cuda()
    cb = cb.cuda()
    parse_cloth = parse_cloth.cuda()
    parse_neck = parse_neck.cuda()
    ya = ya.cuda()
    yb = yb.cuda()
    pa_fake = pa_fake.cuda()
    pb_fake = pb_fake.cuda()
    u = u.cuda()
    grad_outputs = grad_outputs.cuda()

    # Optimizers
    optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # D, optimizerD = amp.initialize(D, optimizerD, opt_level="O1")
    # G, optimizerG = amp.initialize(G, optimizerG, opt_level="O1")

    # Fitting model
    step_start_time = time.time()
    for step in range(opt.n_iter):
        ########################
        # (1) Update D network #
        ########################

        for p in D.parameters():
            p.requires_grad = True

        for t in range(opt.Diters):
            D.zero_grad()

            inputs = train_loader.next_batch()
            with torch.no_grad():
                pa.data.resize_as_(inputs['image']).copy_(inputs['image'])
                ap.data.resize_as_(inputs['agnostic']).copy_(inputs['agnostic'])
                cb.data.resize_as_(inputs['another_cloth']).copy_(inputs['another_cloth'])
            del inputs

            current_batch_size = pa.size(0)
            ya_pred = D(pa)
            _, fake = G(cb, ap)
            with torch.no_grad():
                pb_fake.data.resize_(fake.data.size()).copy_(fake.data)
            del fake
            # Detach y_pred_fake from the neural network G and put it inside D
            with torch.no_grad():
                yb_pred_fake = D(pb_fake.detach())
                ya.resize_(current_batch_size).fill_(1)
                yb.resize_(current_batch_size).fill_(0)

            errD = (BCE_stable(ya_pred - torch.mean(yb_pred_fake), ya) +
                    BCE_stable(yb_pred_fake - torch.mean(ya_pred), yb)) / 2.0
            # with amp.scale_loss(errD, optimizerD) as scaled_loss:
            #     scaled_loss.backward()
            errD.backward()

            # Gradient penalty
            with torch.no_grad():
                u.data.resize_(current_batch_size, 1, 1, 1)
                grad_outputs.resize_(current_batch_size)
            x_both = pa.data * u + pb_fake.data * (1. - u).cuda()
            # We only want the gradients with respect to x_both
            x_both = Variable(x_both, requires_grad=True)
            grad = torch.autograd.grad(outputs=D(x_both), inputs=x_both,
                                       grad_outputs=grad_outputs, retain_graph=True,
                                       create_graph=True, only_inputs=True)[0]
            # We need to norm 3 times (over n_colors x image_size x image_size) to get only a vector of size
            # "batch_size"
            grad_penalty = opt.penalty * ((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
            # with amp.scale_loss(grad_penalty, optimizerD) as scaled_loss:
            #     scaled_loss.backward()
            grad_penalty.backward()

            optimizerD.step()

        ########################
        # (2) Update G network #
        ########################

        for p in D.parameters():
            p.requires_grad = False

        for t in range(opt.Giters):
            G.zero_grad()

            inputs = train_loader.next_batch()
            with torch.no_grad():
                pa.data.resize_as_(inputs['image']).copy_(inputs['image'])
                ap.data.resize_as_(inputs['agnostic']).copy_(inputs['agnostic'])
                cb.data.resize_as_(inputs['another_cloth']).copy_(inputs['another_cloth'])
                ca.data.resize_as_(inputs['cloth']).copy_(inputs['cloth'])
                parse_cloth.resize_as_(inputs['parse_cloth']).copy_(inputs['parse_cloth'])
                parse_neck.resize_as_(inputs['parse_neck']).copy_(inputs['parse_neck'])
            del inputs

            current_batch_size = pa.size(0)

            grid_a, fake = G(ca, ap)
            pa_fake.data.resize_(fake.data.size()).copy_(fake.data)
            del fake
            warped_cloth_a = F.grid_sample(ca, grid_a, padding_mode='border')
            grid_b, fake = G(cb, ap)
            pb_fake.data.resize_(fake.data.size()).copy_(fake.data)
            del fake
            warped_cloth_b = F.grid_sample(cb, grid_b, padding_mode='border')

            ya_pred = D(pa)
            yb_pred_fake = D(pb_fake)
            ya.resize_(current_batch_size).fill_(1)
            yb.resize_(current_batch_size).fill_(0)

            visuals = [
                [cb, warped_cloth_b, pb_fake],
                [ca, warped_cloth_a, pa_fake],
                [ap, parse_cloth, pa]
                # [cb, (warped_cloth_b + pa) * 0.5, pa],
                # [ca, warped_cloth_a, parse_cloth],
                # [parse_neck, (warped_cloth_a + pa) * 0.5, pa]
            ]

            # Non-saturating
            l_adv = 0.1 * (BCE_stable(ya_pred - torch.mean(yb_pred_fake), yb) +
                    BCE_stable(yb_pred_fake - torch.mean(ya_pred), ya)) / 2
            l_warp = criterionWarp(warped_cloth_a, parse_cloth)
            l_perceptual = criterionPerceptual(pa_fake, pa)
            l_L1 = criterionL1(pa_fake, pa)
            loss = l_warp + l_perceptual + l_L1 + l_adv

            optimizerG.zero_grad()
            # with amp.scale_loss(loss, optimizerG) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            optimizerG.step()

            if (step + 1) % opt.display_count == 0:
                board_add_images(board, 'combine', visuals, step + 1)
                board.add_scalar('l_warp', l_warp.item(), step + 1)
                board.add_scalar('l_perceptual', l_perceptual.item(), step + 1)
                board.add_scalar('l_L1', l_L1.item(), step + 1)
                board.add_scalar('l_adv', l_adv.item(), step + 1)
                board.add_scalar('errD', errD.item(), step + 1)
                board.add_scalar('loss', loss.item(), step + 1)
                t = time.time() - step_start_time
                print('step: %8d, time: %.3f, loss: %4f, l_warp: %.4f, l_vgg: %.4f, l_L1: %.4f, l_adv: %.4f, errD: %.4f'
                      % (step + 1, t, loss.item(), l_warp.item(), l_perceptual.item(), l_L1.item(), l_adv.item(), errD.item()),
                      flush=True)
                step_start_time = time.time()

            if (step + 1) % opt.save_count == 0:
                save_checkpoint(G, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


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
    # G = nn.DataParallel(WUTON(opt), device_ids=[0, 1, 2, 3])
    # D = nn.DataParallel(Discriminator(), device_ids=[0, 1, 2, 3])
    G = WUTON(opt)
    D = Discriminator()
    if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):  # TODO
        load_checkpoint(G, opt.checkpoint)
    train(opt, train_loader, G, D, board)
    save_checkpoint(G, os.path.join(opt.checkpoint_dir, opt.name, 'wuton_final.pth'))

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
