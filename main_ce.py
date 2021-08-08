from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
import pickle

import numpy as np
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from networks.resnet_big import SupCEResNet
from ood_util import set_ood_loader, ood_conf
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from vit.src.data_loaders import create_dataloaders
from vit.src.model import VisionTransformer as ViT
from vit.src.utils import write_json

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD',
                        choices=['LARS', 'SGD', 'RMSprop'], help='optimizer')
    parser.add_argument("--image-size", type=int, default=32, help="input image size", choices=[32, 48, 96, 128, 160, 224, 384])
    parser.add_argument('--learning_rate', type=float, default=0.8,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--ood', action='store_true',
                        help='validate out of distribution')
    parser.add_argument('--reduce_lr', type=float, default=0.0,
                        # todo donot change this for contrastive training, 0 ignores reduce_lr
                        help='reduce learning rate for detector')
    parser.add_argument('--contrastive', action='store_true', help='using distributed loss calculations across multiple GPUs')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--use_subset', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet', 'stl10'], help='dataset')
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument('--data_folder', default='./datasets/', type=str)
    parser.add_argument("--data-dir", type=str, default='./datasets/', help='data folder')
    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')



    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--root_folder', default='.', type=str)

    opt = parser.parse_args()

    # set the path according to the environment
    #opt.data_folder = './datasets/'
    opt.model_path = opt.root_folder + '/save/SupCE/{}_models'.format(opt.dataset)
    opt.tb_path = opt.root_folder + '/save/SupCE/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_seed_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial, opt.seed)

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


    # now open and wrtie parameter into file
    to_write = open(os.path.join(opt.save_folder, 'param.txt'), 'w')
    print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
    for x, y in vars(opt).items():
        print("{} : {}".format(x, y))
        to_write.write(str(x) + ' >>> ' + str(y) + '\n')
    to_write.close()

    return opt


def set_loader(opt, img_size=32):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.5, 0.5, 0.5) if opt.model=='vit' else (0.4914, 0.4822, 0.4465)
        std = (0.5, 0.5, 0.5) if opt.model=='vit' else  (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5, 0.5, 0.5) if opt.model=='vit' else (0.5071, 0.4867, 0.4408)
        std = (0.5, 0.5, 0.5) if opt.model=='vit' else (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None

    if opt.use_subset:
        print("Currently using CIFAR 10 subset")
        cls_idx = [j for j,cls in enumerate(train_dataset.classes) if cls in ['airplane', 'automobile','bird', 'cat']]
        train_idx = [i for i, x in enumerate(train_dataset.targets) if x in cls_idx]
        val_idx = [i for i, x in enumerate(val_dataset.targets) if x in cls_idx]
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset if opt.use_subset else train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_subset if opt.use_subset else val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    if opt.model=='vit':
        model = ViT(image_size=(opt.image_size, opt.image_size),
             patch_size=(opt.patch_size, opt.patch_size),
             emb_dim=opt.emb_dim,
             mlp_dim=opt.mlp_dim,
             num_heads=opt.num_heads,
             num_layers=opt.num_layers,
             num_classes=opt.num_classes,
             attn_dropout_rate=opt.attn_dropout_rate,
             dropout_rate=opt.dropout_rate)
    else:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt, in_out_idx):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #for ood
    top_ood = AverageMeter()
    in_conf = AverageMeter()
    out_conf = AverageMeter()

    with torch.no_grad():
        end = time.time()
        max_len = len(val_loader.dataset)
        batch_size = val_loader.batch_size
        for idx, (images, labels) in enumerate(val_loader):

            # sanity check
            if opt.ood:
                index_list = list(range(idx * batch_size, min(((idx * batch_size) + batch_size), max_len)))
                target_array = np.asarray(val_loader.dataset.targets)
                assert (target_array[index_list] == labels.data.numpy()).all()
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            if opt.ood:
                in_acc1, in_score, out_score, in_len, out_len = ood_conf(output, labels, in_out_idx[index_list])
                top_ood.update(in_acc1[0][0], in_len)
                in_conf.update(in_score, in_len)
                out_conf.update(out_score, out_len)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                if opt.ood:
                   print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'In_Acc@1 {top_ood.val:.3f} ({top_ood.avg:.3f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'In score {in_score.val:.3f}\t'
                          'Out score {out_score.val:.3f}\t'
                       .format(
                        idx, len(val_loader), batch_time=batch_time,
                        loss=losses, top_ood=top_ood, top1=top1, in_score=in_conf, out_score=out_conf))
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time,
                           loss=losses, top1=top1))
    if opt.ood:
        print(' * Acc@1 {top1.avg:.3f}\t in acc {i_acc.avg:.3f}\t in score {in_score.avg:.3f}\t out score {out_score.avg:.3f}'.format(
                top1=top1, i_acc=top_ood, in_score=in_conf, out_score=out_conf))

    else:
        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    if opt.seed !=None: # fix the seed for reproducibility
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    # build data loader
    if opt.ood:
        train_loader, val_loader, out_index = set_ood_loader(opt)
    else:
        train_loader, val_loader = create_dataloaders(opt)
        out_index = None

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    params = [p for n, p in (list(model.named_parameters())) if p.requires_grad]
    optimizer = set_optimizer(opt, params)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # saving the model before training (epoch 0)
    epoch = 0
    save_file = os.path.join(
        opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
    save_model(model, optimizer, opt, epoch, save_file)

    val_acc_list = []
    save_epoch_list = []
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('train_loss', loss, epoch)
        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluation
        loss, val_acc = validate(val_loader, model, criterion, opt, out_index)
        logger.log_value('val_loss', loss, epoch)
        logger.log_value('val_acc', val_acc, epoch)

        val_acc_list.append(val_acc)
        save_epoch_list.append(epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_best.pth')
            save_model(model, optimizer, opt, epoch, save_file)


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)



    # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))
    best_curr_acc = {'acc_list': val_acc_list, 'save_epoch': save_epoch_list, }
    write_json(best_curr_acc, os.path.join(opt.save_folder, 'acc.json'))
    pickle_obj = {'Test_acc': val_acc_list, 'Epoch': save_epoch_list}
    fname = os.path.join(opt.save_folder, 'accuracy.pickle')
    with open(fname, 'wb') as f1:
        pickle.dump(pickle_obj, f1)


if __name__ == '__main__':
    run_on_all_data = False  # todo for running all the data dataset
    if run_on_all_data:
        import sys
        dataset_list = ['cifar10', 'cifar100', 'ImageNet']
        for dataset in dataset_list:
            index_of_ds = sys.argv.index('--dataset') + 1
            index_ds_cls = sys.argv.index('--num-classes') + 1
            if dataset == 'cifar10':
                sys.argv[index_of_ds] = 'cifar10'
                sys.argv[index_ds_cls] = str(10)
            elif dataset == 'cifar100':
                sys.argv[index_of_ds] = 'cifar100'
                sys.argv[index_ds_cls] = str(100)
            else:
                sys.argv[index_of_ds] = 'ImageNet'
                sys.argv[index_ds_cls] = str(30)
                sys.argv[sys.argv.index('--image-size')+1] = str(224)
            main()
    else:
        main()
