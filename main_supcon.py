from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import copy
import random

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from main_ce import set_loader as set_val_loader
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier
from losses import SupConLoss
from parallel import DataParallelModel, DataParallelCriterion
from main_linear import validate
from main_linear import train as train_classifier
from vit.src.model import VisionTransformer as ViT
from vit.src.utils import setup_device
from vit.src.checkpoint import load_checkpoint
from vit.src.data_loaders import create_dataloaders

from vit.src.config import *


#from pytorch_pretrained_vit import ViT, load_pretrained_weights


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"# 0,1,2,3,5,6,7
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=700,
                        help='number of training epochs')
    parser.add_argument('--val_freq', type=int, default=50,
                        help='number of epochs after which validation is done/model saved')

    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD',    # CT with simclr default with 'LARS' optimizer, momentum 0.9 and weight decay 1e-6 and initial LR 1.0
                        choices=['LARS', 'SGD', 'RMSprop'], help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.5, # CT with simclr default: 1.0
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='500,700,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, # CT with simclr default: 1e-6
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,     # CT with simclr default: 0.9
                        help='momentum')
    parser.add_argument('--reduce_lr', type=float, default=0.0,     #todo donot change this for contrastive training, 0 ignores reduce_lr
                        help='reduce learning rate for detector')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50',
                choices=['resnet18','resnet34','resnet50','resnet200', 'wide_resnet50_3', 'vit'], help='network type')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet','imagenet30','stl10'], help='dataset')
    parser.add_argument('--use_subset', action='store_true', help='sub set of classes')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    # similarity metric
    parser.add_argument('--sim_metric', type=str, default='Cosine',
                        choices=['Cosine', 'Euclidean', 'Mahalanobis'], help='similarity metric used in contrastive loss')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,  # CT with simclr default: 0.1
                        help='temperature for loss function')
    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')


    # other setting
    parser.add_argument('--distributed', action='store_true',
                        help='using distributed loss calculations across multiple GPUs')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--half', action='store_true',
                        help='train using half precision')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--seed', default=12321, type=int)
    parser.add_argument('--data_folder', default='./datasets/', type=str)
    parser.add_argument('--root_folder', default='.', type=str)

    #vit related settings
    parser.add_argument("--model_arch", type=str, default="b16", help='model setting to use',
                        choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=32, help="input image size", choices=[32, 48, 96, 224, 384, 160])
    parser.add_argument("--train-steps", type=int, default=10000, help="number of training/fine-tunning steps")


    opt = parser.parse_args()
    if opt.model=='vit':
        # model config
        opt = eval("get_{}_config".format(opt.model_arch))(opt)
        #process_config(opt)


    # set the path according to the environment
    #opt.data_folder = './datasets/'
    opt.model_path = opt.root_folder + '/save/{}/{}_models'.format(opt.method, opt.dataset)
    opt.tb_path = opt.root_folder + '/save/{}/{}_tensorboard'.format(opt.method, opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_dist_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model if opt.model!='vit' else opt.model+'_'+opt.model_arch, opt.learning_rate,
               opt.sim_metric,opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 30         # CT with simclr default: 30
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
    to_write = open(os.path.join(opt.save_folder,'param.txt'), 'w')
    print("~~~~~~~~ Hyperparameters used: ~~~~~~~")
    for x, y in vars(opt).items():
        print("{} : {}".format(x, y))
        to_write.write(str(x) + ' >>> ' + str(y) + '\n')
    to_write.close()

    if opt.dataset == 'cifar10' or opt.dataset == 'stl10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset == 'imagenet30':
        opt.n_cls = 30
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    # for main linear in validation
    val_opt = copy.deepcopy(opt)
    val_opt.ood = False
    val_opt.train_type = "single"
    val_opt.save_t_SNE = False
    val_opt.learning_rate = 0.001
    val_opt.train_model =False

    return opt, val_opt


def set_loader(opt):
    opt.data_dir = opt.data_folder
    opt.contrastive = True
    train_loader, val_loader = create_dataloaders(opt)
    val_train_loader = val_val_loader =None
    # scale = (0.2, 1.)
    # # construct data loader
    # if opt.dataset == 'cifar10':
    #     mean = (0.4914, 0.4822, 0.4465)
    #     std = (0.2023, 0.1994, 0.2010)
    # elif opt.dataset == 'cifar100':
    #     mean = (0.5071, 0.4867, 0.4408)
    #     std = (0.2675, 0.2565, 0.2761)
    # elif 'imagenet' in opt.dataset or opt.model=='vit':
    #     mean =(0.485, 0.456, 0.406)
    #     std = (0.229, 0.224, 0.225)
    #     scale = (0.08, 1.0) #defaul for random resize corp
    # else:
    #     raise ValueError('dataset not supported: {}'.format(opt.dataset))
    # normalize = transforms.Normalize(mean=mean, std=std)
    #
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(size=opt.image_size, scale=scale),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    #     ], p=0.8),
    #     transforms.RandomGrayscale(p=0.2),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    #
    # if opt.dataset == 'cifar10':
    #     train_dataset = datasets.CIFAR10(root=opt.data_folder,
    #                                      transform=TwoCropTransform(train_transform),
    #                                      download=True)
    #     val_dataset = datasets.CIFAR10(root=opt.data_folder,
    #                                    train=False,
    #                                    transform=val_transform)
    # elif opt.dataset == 'cifar100':
    #     train_dataset = datasets.CIFAR100(root=opt.data_folder,
    #                                       transform=TwoCropTransform(train_transform),
    #                                       download=True)
    #     val_dataset = datasets.CIFAR100(root=opt.data_folder,
    #                                    train=False,
    #                                    transform=val_transform)
    # elif 'imagenet' in opt.dataset :
    #     train_dataset = datasets.ImageNet(root=os.path.join(opt.data_folder,'LSVRC2015' if opt.dataset=='imagenet' else 'imagenet30'),
    #                                       transform=TwoCropTransform(train_transform),
    #                                       download=False)
    #     val_dataset = datasets.ImageNet(root=opt.data_folder,
    #                                     train=False,
    #                                     transform=val_transform)
    # else:
    #     raise ValueError(opt.dataset)
    #
    # if opt.use_subset:
    #     print("Currently using CIFAR 10 subset")
    #     cls_idx = [j for j,cls in enumerate(train_dataset.classes) if cls in ['airplane', 'automobile','bird', 'cat']]
    #     train_idx = [i for i, x in enumerate(train_dataset.targets) if x in cls_idx]
    #     val_idx = [i for i, x in enumerate(val_dataset.targets) if x in cls_idx]
    #     train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    #     val_subset = torch.utils.data.Subset(val_dataset, val_idx)
    #
    # train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     train_subset if opt.use_subset else train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
    #     num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    #
    # val_loader = torch.utils.data.DataLoader(
    #     val_subset if opt.use_subset else val_dataset, batch_size=opt.batch_size, shuffle=False,
    #     num_workers=opt.num_workers, pin_memory=True)
    #
    # #call validation loader as supcon loader two corp transform
    # # create dataloader
    # print("create dataloaders for validation")
    # val_train_loader = eval("{}DataLoader".format(opt.dataset))(
    #     data_dir=opt.data_folder,
    #     image_size=opt.image_size,
    #     batch_size=opt.batch_size,
    #     num_workers=opt.num_workers,
    #     split='train')
    # val_val_loader = eval("{}DataLoader".format(opt.dataset))(
    #     data_dir=opt.data_folder,
    #     image_size=opt.image_size,
    #     batch_size=opt.batch_size,
    #     num_workers=opt.num_workers,
    #     split='val')
    # #val_train_loader, val_val_loader = set_val_loader(opt)

    return train_loader, val_loader, val_train_loader, val_val_loader


def set_model(opt):
    if opt.model=='vit':
        model = ViT(image_size=(opt.image_size, opt.image_size),
             patch_size=(opt.patch_size, opt.patch_size),
             emb_dim=opt.emb_dim,
             mlp_dim=opt.mlp_dim,
             num_heads=opt.num_heads,
             num_layers=opt.num_layers,
             num_classes=opt.n_cls,
             attn_dropout_rate=opt.attn_dropout_rate,
             dropout_rate=opt.dropout_rate)
    else:
        model = SupConResNet(name=opt.model)

    if opt.checkpoint_path != None:
        state_dict = load_checkpoint(opt.checkpoint_path, new_img=opt.image_size, patch=opt.patch_size)
        if opt.n_cls != state_dict['classifier.weight'].size(0):
            del state_dict['classifier.weight']
            del state_dict['classifier.bias']
            print("re-initialize fc layer")
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(opt.checkpoint_path))

    if opt.distributed:
        model = DataParallelModel(model)
        criterion = DataParallelCriterion(SupConLoss(temperature=opt.temp, similarity_metric=opt.sim_metric))
        ce_loss = DataParallelCriterion(torch.nn.CrossEntropyLoss())
        classifier = DataParallelModel(LinearClassifier(name=opt.model, num_classes=opt.n_cls))


        model = model.cuda()
        criterion = criterion.cuda()
        ce_loss = ce_loss.cuda()
        classifier.cuda()
        cudnn.benchmark = True

    else:
        criterion = SupConLoss(temperature=opt.temp, similarity_metric=opt.sim_metric)
        ce_loss = torch.nn.CrossEntropyLoss()

        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

        # enable synchronized Batch Normalization
        if opt.syncBN:
            model = apex.parallel.convert_syncbn_model(model)

        if torch.cuda.is_available():
            print('__Number CUDA Devices:', torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                device, device_ids = setup_device(torch.cuda.device_count())
                model = model.to(device)
                classifier = classifier.to(device)
                if len(device_ids) > 1:
                    model = torch.nn.DataParallel(model, device_ids=device_ids)
                    classifier = torch.nn.DataParallel(classifier, device_ids=device_ids)
            else:
                model = model.cuda()
                classifier = classifier.cuda()
                model.encoder = torch.nn.DataParallel(model.encoder if opt.model !='vit' else model) # todo mayb this is not needed
             #todo this is done for vit, check if works for supconn too
            if opt.half:
                model.half()  # convert to half precision
                for layer in model.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        layer.float()
            criterion = criterion.cuda()
            ce_loss = ce_loss.cuda()
            cudnn.benchmark = True

    return model, classifier, criterion, ce_loss


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if isinstance(images[0], dict):  # for albumnetations
            images = torch.cat([images[0]['image'], images[1]['image']], dim=0)  # .to(device)
        else:
            images = torch.cat([images[0], images[1]], dim=0)  # .to(device)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        if opt.half:
            images = images.half()
        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
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

def main():
    opt, val_opt = parse_option()
    if opt.seed !=None: # fix the seed for reproducibility
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    global_best_val_acc = 0

    # build data loader
    train_loader, val_loader, val_train_loader, val_val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion, cross_entropy = set_model(opt)

    # build optimizer
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = set_optimizer(opt, params)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # saving the model before training (epoch 0)
    epoch = 0
    save_file = os.path.join(
        opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
    print(save_file)
    save_model(model, optimizer, opt, epoch, save_file)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, train loss {:.4f}'.format(epoch, time2 - time1, loss))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # validation module at fixed interval of epochs
        if epoch % opt.val_freq == 0: # reset validation param and optimizer
            # local_best_val_acc = 0
            # local_best_val_loss = 0
            # torch.nn.init.zeros_(classifier.module.fc.weight) # reinit weight every time
            # val_optimizer = set_optimizer(val_opt, [p for n, p in classifier.named_parameters() if p.requires_grad])
            #
            # for val_epoch in range(0, 1):  #train classifier for few epoch
            #
            #     _ = train_classifier(val_train_loader, model, classifier, cross_entropy, None,val_optimizer, val_epoch, val_opt, val_supcon=True)
            #
            #     val_loss, val_acc, _ = validate(val_val_loader, model, classifier, cross_entropy, val_opt, supcon_val=True)
            #
            # if val_acc > local_best_val_acc:
            #     local_best_val_acc = val_acc
            #     local_best_val_loss = val_loss
            # print('epoch {}, val acc {:.4f}, val loss {:.4f}'.format(epoch, val_acc, val_loss))
            # logger.log_value('val_acc', val_acc, epoch)
            # logger.log_value('val_loss', local_best_val_loss, epoch)
            #
            # if local_best_val_acc>global_best_val_acc:
            #     global_best_val_acc = local_best_val_acc
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            print(save_file)
            save_model(model, optimizer, opt, epoch, save_file)


    # save the last model
    # save_file = os.path.join(
    #     opt.save_folder, 'last.pth')
    # save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best validation accuracy: {:.2f}'.format(global_best_val_acc))

if __name__ == '__main__':
    main()
