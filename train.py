# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import os
import time
from cp_dataset import CPDataset, CPDataLoader
from networks import VGGLoss, load_checkpoint, save_checkpoint, WUTON, Discriminator, HumanParser
from tensorboardX import SummaryWriter
from visualization import board_add_images
from PIL import Image


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="GMM")
    parser.add_argument("--gpu_ids", default="")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument("--dataroot", default="dataset")
    parser.add_argument("--datamode", default="train")
    parser.add_argument("--stage", default="GMM")
    parser.add_argument("--data_list", default="train.csv")
    parser.add_argument("--fine_width", type=int, default=192)
    parser.add_argument("--fine_height", type=int, default=256)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default=20)
    parser.add_argument("--save_count", type=int, default=5000)
    parser.add_argument("--keep_step", type=int, default=100000)
    parser.add_argument("--decay_step", type=int, default=100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--n_iter", type=int, default=100000, help='Number of iteration cycles')
    parser.add_argument('--Diters', type=int, default=1, help='Number of iterations of D')
    parser.add_argument('--Giters', type=int, default=1, help='Number of iterations of G.')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--penalty', type=float, default=10, help='Gradient penalty parameter for WGAN-GP')
    parser.add_argument("--human_parser_step", type=int, default=0, help='add human parser after some steps')

    opt = parser.parse_args()
    return opt


def get_palette(num_cls=20):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


palette = get_palette()


def visualize_seg(segs):
    # (N,H,W)
    N, H, W = segs.size()
    res = torch.zeros((N, 3, H, W))
    for i, seg in enumerate(segs):
        seg = seg.cpu().numpy()
        output_img = Image.fromarray(np.asarray(seg, dtype=np.uint8))
        output_img.putpalette(palette)
        res[i] = torch.from_numpy(np.array(output_img))
    return res


def train(opt, train_loader, G, D, board):
    human_parser = HumanParser(opt)
    human_parser.eval()
    G.train()
    D.train()

    # palette = get_palette()

    # Criterion
    criterionWarp = nn.L1Loss()
    criterionPerceptual = VGGLoss()
    criterionL1 = nn.L1Loss()
    BCE_stable = nn.BCEWithLogitsLoss()

    # Variables
    pa = torch.FloatTensor((opt.batch_size, 3, opt.fine_height, opt.fine_width))
    ap = torch.FloatTensor((opt.batch_size, 3, opt.fine_height, opt.fine_width))
    ca = torch.FloatTensor((opt.batch_size, 3, opt.fine_height, opt.fine_width))
    cb = torch.FloatTensor((opt.batch_size, 3, opt.fine_height, opt.fine_width))
    parse_cloth = torch.FloatTensor((opt.batch_size, 3, opt.fine_height, opt.fine_width))
    ya = torch.FloatTensor(opt.batch_size)
    yb = torch.FloatTensor(opt.batch_size)
    u = torch.FloatTensor((opt.batch_size, 1, 1, 1))
    grad_outputs = torch.ones(opt.batch_size)

    # Everything cuda
    if opt.cuda:
        G.cuda()
        D.cuda()
        human_parser.cuda()
        criterionWarp = criterionWarp.cuda()
        criterionPerceptual = criterionPerceptual.cuda()
        criterionL1 = criterionL1.cuda()
        BCE_stable.cuda()
        pa = pa.cuda()
        ap = ap.cuda()
        ca = ca.cuda()
        cb = cb.cuda()
        parse_cloth = parse_cloth.cuda()
        ya = ya.cuda()
        yb = yb.cuda()
        u = u.cuda()
        grad_outputs = grad_outputs.cuda()

        # DataParallel
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        human_parser = nn.DataParallel(human_parser)

    # Optimizers
    optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0.5, 0.999))

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
                pa.resize_as_(inputs['image']).copy_(inputs['image'])
                ap.resize_as_(inputs['agnostic']).copy_(inputs['agnostic'])
                cb.resize_as_(inputs['another_cloth']).copy_(inputs['another_cloth'])
            del inputs

            current_batch_size = pa.size(0)
            ya_pred = D(pa)
            _, pb_fake = G(cb, ap)

            # Detach y_pred_fake from the neural network G and put it inside D
            yb_pred_fake = D(pb_fake.detach())
            ya.data.resize_(current_batch_size).fill_(1)
            yb.data.resize_(current_batch_size).fill_(0)

            errD = (BCE_stable(ya_pred - torch.mean(yb_pred_fake), ya) +
                    BCE_stable(yb_pred_fake - torch.mean(ya_pred), yb)) / 2.0
            # with amp.scale_loss(errD, optimizerD) as scaled_loss:
            #     scaled_loss.backward()
            errD.backward()

            # Gradient penalty
            with torch.no_grad():
                u.resize_(current_batch_size, 1, 1, 1).uniform_(0, 1)
                grad_outputs.data.resize_(current_batch_size)
            x_both = pa.data * u + pb_fake.data * (1. - u)
            if opt.cuda:
                x_both = x_both.cuda()
            # We only want the gradients with respect to x_both
            x_both = Variable(x_both, requires_grad=True)
            grad = torch.autograd.grad(outputs=D(x_both), inputs=x_both,
                                       grad_outputs=grad_outputs, retain_graph=True,
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

        for t in range(opt.Giters):
            inputs = train_loader.next_batch()
            with torch.no_grad():
                pa.resize_as_(inputs['image']).copy_(inputs['image'])
                ap.resize_as_(inputs['agnostic']).copy_(inputs['agnostic'])
                ca.resize_as_(inputs['cloth']).copy_(inputs['cloth'])
                cb.resize_as_(inputs['another_cloth']).copy_(inputs['another_cloth'])
                parse_cloth.resize_as_(inputs['parse_cloth']).copy_(inputs['parse_cloth'])
            del inputs

            current_batch_size = pa.size(0)

            # paired data
            G.zero_grad()

            warped_cloth_a, pa_fake = G(ca, ap)
            if step >= opt.human_parser_step:  # 生成的图片较真实后再添加human parser
                parse_pa_fake = human_parser(pa_fake)  # (N,H,W)
                parse_ca_fake = (parse_pa_fake == 5) + \
                                (parse_pa_fake == 6) + \
                                (parse_pa_fake == 7)  # [0,1] (N,H,W)
                parse_ca_fake = parse_ca_fake.unsqueeze(1).type_as(pa_fake)  # (N,1,H,W)
                ca_fake = pa_fake * parse_ca_fake + (1 - parse_ca_fake)  # [-1,1]
                parse_pa_fake = visualize_seg(parse_pa_fake)
                if opt.cuda:
                    parse_pa_fake = parse_pa_fake.cuda()
            else:
                ca_fake = torch.zeros_like(pa_fake)
                parse_pa_fake = torch.zeros_like(pa_fake)

            l_warp = criterionWarp(warped_cloth_a, parse_cloth)
            l_perceptual = criterionPerceptual(pa_fake, pa)
            l_L1 = criterionL1(pa_fake, pa)
            loss = l_warp + l_perceptual + l_L1

            loss.backward()
            optimizerG.step()

            # unpaired data
            G.zero_grad()

            warped_cloth_b, pb_fake = G(cb, ap)
            if step >= opt.human_parser_step:  # 生成的图片较真实后再添加human parser
                parse_pb_fake = human_parser(pb_fake)
                parse_cb_fake = (parse_pb_fake == 5) + \
                                (parse_pb_fake == 6) + \
                                (parse_pb_fake == 7)  # [0,1] (N,H,W)
                parse_cb_fake = parse_cb_fake.unsqueeze(1).type_as(pb_fake)  # (N,1,H,W)
                cb_fake = pb_fake * parse_cb_fake + (1 - parse_cb_fake)  # [-1,1]
                parse_pb_fake = visualize_seg(parse_pb_fake)
                if opt.cuda:
                    parse_pb_fake = parse_pb_fake.cuda()
            else:
                cb_fake = torch.zeros_like(pb_fake)
                parse_pb_fake = torch.zeros_like(pb_fake)

            with torch.no_grad():
                ya.data.resize_(current_batch_size).fill_(1)
                yb.data.resize_(current_batch_size).fill_(0)
            ya_pred = D(pa)
            yb_pred_fake = D(pb_fake)

            # Non-saturating
            l_adv = (BCE_stable(ya_pred - torch.mean(yb_pred_fake), yb) +
                     BCE_stable(yb_pred_fake - torch.mean(ya_pred), ya)) / 2

            l_adv.backward()
            optimizerG.step()

            # visuals = [
            #     [cb, warped_cloth_b, pb_fake],
            #     [ca, warped_cloth_a, pa_fake],
            #     [ap, parse_cloth, pa]
            # ]
            visuals = [
                [cb, warped_cloth_b, pb_fake, cb_fake, parse_pb_fake],
                [ca, warped_cloth_a, pa_fake, ca_fake, parse_pa_fake],
                [ap, parse_cloth, pa]
            ]

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
                      % (step + 1, t, loss.item(), l_warp.item(), l_perceptual.item(), l_L1.item(), l_adv.item(),
                         errD.item()),
                      flush=True)
                step_start_time = time.time()

            if (step + 1) % opt.save_count == 0:
                save_checkpoint(G, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))


def main():
    opt = get_opt()
    opt.cuda = False
    opt.batch_size = 1
    opt.name = "test"
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
    # train2(opt, train_loader, G, board)
    save_checkpoint(G, os.path.join(opt.checkpoint_dir, opt.name, 'wuton_final.pth'))

    print('Finished training %s, named: %s!' % (opt.stage, opt.name))


if __name__ == "__main__":
    main()
