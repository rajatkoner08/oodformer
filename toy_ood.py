
from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import numpy as np


from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from ood_util import ood_conf
from util import set_optimizer, save_model
from networks.toy_net_and_dataset import *
from losses import SupConLoss


import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1, 2"


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=5,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--model', type=str, default='ToyNet')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--ood', action='store_true',
                        help='validate out of distribution')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'MainCE'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    #checkpoint
    parser.add_argument('--ckpt', type=str, default='',help='path to pre-trained model')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')


    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/{}/toy_models'.format(opt.method)
    opt.tb_path = './save/{}/toy_tensorboard'.format(opt.method)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}'.\
        format(opt.method, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):

    """
    train_transform = transforms.Compose([
        transforms.RandomApply([
        AddGaussianNoise(0., 1.)], p=0.8),
        transforms.ToTensor(),
    ])
    """

    train_dataset = ToyDataset()
    val_dataset = ToyDataset(samples=1000, ood=opt.ood)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True,)

    return train_loader, val_loader

def set_model(opt):
    model = ToyNet(head=opt.method)
    if opt.method=='mainCE':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = SupConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('Device name :',torch.cuda.current_device())
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    #load model
    if not opt.ckpt == '':
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)

    return model, criterion


def calc_mean_cov(model,num_classes, feature_list, train_loader):
    print('TBD')



def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    if opt.method =='MainCE':
        model.feature_extractor.eval()
        model.head.train()
    else:
        model.feature_extractor.train()
        model.head.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (point_in2d, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        #fixme add twoDataTransfrom
        if opt.method in ("SupCon","SimCLR"):
            point_in2d = torch.cat([point_in2d[0], point_in2d[1]], dim=0)
        point_in2d = point_in2d.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        main_emb, head_emb = model(point_in2d)
        if opt.method in ("SupCon", "SimCLR"):
            f1, f2 = torch.split(head_emb, [bsz, bsz], dim=0)
            head_emb = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method == 'SupCon' or opt.method == 'MainCE':
            loss = criterion(head_emb, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(head_emb)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def validate(val_loader, model, criterion, opt, in_out_idx):
    """validation"""
    model.feature_extractor.eval()
    model.head.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top_ood = AverageMeter()
    in_conf = AverageMeter()
    out_conf = AverageMeter()

    with torch.no_grad():
        end = time.time()

        feats_emb = []

        for idx, (points_in2d, labels) in enumerate(val_loader):
            bsz = labels.shape[0]

            points_in2d = points_in2d.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # forward
            main_emb, head_emb = model(points_in2d)
            loss = criterion(head_emb, labels)
            if len(feats_emb) == 0:
                feats_emb = main_emb
            else:
                feats_emb = torch.cat((feats_emb, main_emb))

            #fixme for acc exclude ood, as no label is there
            # update metric
            losses.update(loss.item(), bsz)
            acc = binary_acc(head_emb, labels)
            top1.update(acc[0], bsz)

            # if opt.ood:
            #     in_acc1, in_score, out_score, in_len, out_len = ood_conf(head_emb, labels, in_out_idx[index_list])
            #     top_ood.update(in_acc1[0][0], in_len)
            #     in_conf.update(in_score, in_len)
            #     out_conf.update(out_score, out_len)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                   print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       .format(idx, len(val_loader), batch_time=batch_time,
                        loss=losses, top1=top1))

    return losses.avg, top1.avg, feats_emb



def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)


    # Calculate mean and covariance of train data
    mean, cov = calc_mean_cov(model, num_classes=2, feature_list=[16, 32], train_loader)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('Train : epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if opt.model=='mainCE':
            # eval for one epoch
            val_loss, val_acc, resnet_emb = validate(val_loader, model, criterion, opt)
            print('Validation : epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc

        # writer.add_scalar('linear_train_loss', loss, epoch)
        # writer.add_scalar('linear_val_loss', loss, epoch)
        # writer.add_scalar('linear_val_acc', val_acc, epoch)
        # writer.add_scalar('linear_val_loss', val_loss, epoch)

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            print(save_file)
            save_model(model, optimizer, opt, epoch, save_file)
    if opt.model == 'mainCE':
        print('Best Acc : ',best_acc)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()