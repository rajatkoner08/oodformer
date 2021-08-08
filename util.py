from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
#from lars import LARS                 # Install additional package using: pip install torchlars
from another_lars import LARS
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
from pathlib import Path

def write_pickle(content, fname):
    fname = Path(fname)
    with fname.open('wb') as handle:
        pickle.dump(content, handle)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, albumentations=False):
        self.transform = transform
        self.albumentations = albumentations

    def __call__(self, x):
        if self.albumentations:
            return [self.transform(image=np.array(x)), self.transform(image=np.array(x))]
        else:
            return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    args.lr_decay_rate = 0.01
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        print("Eta mean : ",eta_min," cos ",(
                1 + math.cos(math.pi * epoch / args.epochs)) / 2)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        if 0.0 <args.reduce_lr <1.0:
            i = 0
            for param_group in optimizer.param_groups:  # the orderring is imp, as per the odering of reduce lr
                if i == 0:
                    param_group['lr'] = lr * args.reduce_lr
                else:
                    param_group['lr'] = lr
                i += 1
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def set_optimizer(opt, params):
    base_optimizer = optim.SGD(params=params,
                               lr=opt.learning_rate,
                               momentum=opt.momentum,
                               weight_decay=opt.weight_decay)
    if opt.optimizer == 'SGD':
        optimizer = base_optimizer
    elif opt.optimizer == 'LARS':
        optimizer = LARS(params=params,
                         lr=opt.learning_rate,
                         momentum=opt.momentum,
                         weight_decay=opt.weight_decay,
                         )#LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    elif opt.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(params,
                                  lr=opt.learning_rate,
                                  momentum=opt.momentum,
                                  weight_decay=opt.weight_decay)

    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, coeff_smooth):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = coeff_smooth

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()