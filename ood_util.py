from torchvision import transforms, datasets
import torch
import numpy as np
import sys
import random
import sklearn.metrics as skm
from util import *

#OOD validation loader
def set_ood_loader(opt, out_in_ratio=0.2, out_two_demo=True, image_size=32):
    ood_dataset = ''

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.5, 0.5, 0.5) if opt.model=='vit' else (0.4914, 0.4822, 0.4465)
        std =  (0.5, 0.5, 0.5) if opt.model=='vit' else (0.2023, 0.1994, 0.2010)
        ood_dataset = 'cifar100'
        crop_size = 32
    elif opt.dataset == 'cifar100':
        mean = (0.5, 0.5, 0.5) if opt.model=='vit' else (0.5071, 0.4867, 0.4408)
        std = (0.5, 0.5, 0.5) if opt.model=='vit' else (0.2675, 0.2565, 0.2761)
        ood_dataset = 'cifar10'
        crop_size = 32
    elif opt.dataset == 'stl10':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        ood_dataset = 'cifar10'
        crop_size = 96
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)          # Added TwoCropTransform for CT using contrastive loss in main_linear
        val_in_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
        val_out_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)         # Added TwoCropTransform for CT using contrastive loss in main_linear
        val_in_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
        val_out_dataset = datasets.CIFAR10(root=opt.data_folder,
                                          train=False,
                                          transform=val_transform)

    elif opt.dataset == 'stl10':
        train_dataset = datasets.STL10(root=opt.data_folder,
                                       split='train+unlabeled',
                                       transform=TwoCropTransform(train_transform),
                                       download=True)         # Added TwoCropTransform for CT using contrastive loss in main_linear
        val_in_dataset = datasets.STL10(root=opt.data_folder,
                                        split='test',
                                        train=False,
                                        transform=val_transform)
        val_out_dataset = datasets.CIFAR10(root=opt.data_folder,
                                          train=False,
                                          transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    dataset_len = len(val_in_dataset.targets)
    assert dataset_len == len(val_out_dataset.targets)
    if out_two_demo and opt.dataset=='cifar10': #demo vis for two out classes, one close one far
        near_cls = 58 #pickup truck
        far_cls = 82 #sunflower
        near_out_index = np.where(np.asarray(val_out_dataset.targets)==near_cls)
        far_out_index = np.where(np.asarray(val_out_dataset.targets) == far_cls)
        out_index = np.where(np.isin(np.asarray(val_out_dataset.targets), [near_cls, far_cls]))
    elif opt.dataset=='cifar10':
        out_index = random.sample(range(0, dataset_len), int(dataset_len*out_in_ratio))
    else:
        print('Only CFIAR10 is supported')

    #copy out dist data to in dist
    val_in_dataset.data[out_index, :, :, :] = val_out_dataset.data[out_index,:,:,:]
    in_out_idx = np.zeros(len(val_in_dataset.targets),dtype=int)
    if out_two_demo:
        in_out_idx[near_out_index] = 1
        in_out_idx[far_out_index] = 2
    else:
        in_out_idx[out_index] =1

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_in_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader, in_out_idx


def ood_conf(output, target, in_out_index, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        #for ood only top 1
        assert len(topk)==1 and topk[0]==1


        maxk = max(topk)
        batch_size = target.size(0)

        score, pred = F.softmax(output, dim=1).topk(maxk, 1, True, True)
        pred = pred[np.where(1-in_out_index)].t()
        correct = pred.eq(target[np.where(1-in_out_index)].view(1, -1).expand_as(pred))

        in_dist_score = score.view(-1)[np.where(1-in_out_index)]
        out_dist_score = score.view(-1)[np.where(in_out_index)]

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res, torch.mean(in_dist_score), torch.mean(out_dist_score), len(in_dist_score), len(out_dist_score)

def custom_set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        ood_dataset = 'cifar100'
        crop_size = 32
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        ood_dataset = 'cifar10'
        crop_size = 32
    elif opt.dataset == 'stl10':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        ood_dataset = 'cifar10'
        crop_size = 96
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)         # Added TwoCropTransform for CT using contrastive loss in main_linear
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)        # Added TwoCropTransform for CT using contrastive loss in main_linear
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)

    elif opt.dataset == 'stl10':
        train_dataset = datasets.STL10(root=opt.data_folder,
                                       split='train+unlabeled',
                                       transform=TwoCropTransform(train_transform),
                                       download=True)         # Added TwoCropTransform for CT using contrastive loss in main_linear
        val_dataset = datasets.STL10(root=opt.data_folder,
                                        split='test',
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


#### OOD detection ####
def get_roc_sklearn(xin, xood):
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc