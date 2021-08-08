from __future__ import print_function

import sys
import argparse
import time
import math
import numpy as np
import os
import pickle

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from main_ce import set_loader
from ood_util import set_ood_loader,custom_set_loader,ood_conf
from util import write_json, write_pickle
from util import adjust_learning_rate, warmup_learning_rate, accuracy, AverageMeter
from util import set_optimizer, save_model
from util import LabelSmoothingCrossEntropy

from networks.resnet_big import SupConResNet, LinearClassifier
from losses import SupConLoss
from vit.src.model import VisionTransformer as ViT
from vit.src.config import *
from vit.src.data_loaders import create_dataloaders


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

#os.environ["CUDA_VISIBLE_DEVICES"]="0"
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--save_t_SNE', type=int, default=1000,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=5,     # CT with simclr default: 100
                        help='number of training epochs')

    # optimization
    parser.add_argument('--optimizer', type=str, default='RMSprop',# CT with simclr default with 'LARS' optimizer, momentum 0.9 and weight decay 1e-6 and initial LR 1.0
                        choices=['LARS', 'SGD', 'RMSprop'], help='optimizer')
    parser.add_argument('--learning_rate', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--reduce_lr', type=float, default=0,
                        help='reduce learning rate for detector')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50', # CT with simclr use 'wide_resnet50_3'
                        choices=['resnet50', 'resnet34', 'resnet200', 'wide_resnet50_3','vit'], help='network type')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'stl10', 'cifar100'], help='dataset')
    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    # loss coefficient
    parser.add_argument('--lambda_sup', type=float, default=100, # CT with simclr default: 100
                        help='weight for the softmax CE loss')
    # label smoothing
    parser.add_argument('--coeff_smooth', type=float, default=0.1, # CT with simclr default: 0.1
                        help='coefficient for label smoothing')
    # other setting
    parser.add_argument('--test_contrastive_acc', action='store_true',
                        help='test iterative accuracy accross all epoch for contrastive loss')
    parser.add_argument('--train_model', action='store_true',
                        help='train the object detecetion model')
    parser.add_argument('--train_type', type=str, default='single',
                        choices=['single', 'both'], help='Single indicates either sup/unsup training. Both indicates initial unsup training with two heads(mlp projection, linear mapping)')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--half', action='store_true',
                        help='train using half precision')
    parser.add_argument('--ood', action='store_true',
                        help='validate out of distribution')
    parser.add_argument('--use_subset', action='store_true', help='sub set of classes')

    parser.add_argument('--ood_two_cls_vis', action='store_true',
                        help='visualization of two class(near, far) in ood')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to pre-trained model')
    parser.add_argument('--ckpt_cl', type=str,
                        #default='./save/SupCon/cifar10_models/SupCon_cifar10_resnet50_lr_0.5_decay_0.0001_bsz_512_temp_0.1_trial_0_cosine_warm/ckpt_cl_epoch_1.pth',
                        help='for testing model wd full params')
    parser.add_argument('--cl_save_folder', type=str, default=None, help='path to save classifier ckpt should be saved in ckpt folder')
    parser.add_argument("--model_arch", type=str, default="b16", help='model setting to use',
                        choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--image-size", type=int, default=32, help="input image size", choices=[32, 48, 96, 224, 384])

    parser.add_argument('--data_folder', default='./datasets/', type=str)
    parser.add_argument('--check_ood_cluster', action='store_true', help='to check for default clusters in ood embeddings')
    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')

    opt = parser.parse_args()
    if opt.model=='vit':
        # model config
        opt = eval("get_{}_config".format(opt.model_arch))(opt)

    # set the path according to the environment
    #opt.data_folder = './datasets/'

    #opt.tb_path = './save/Main_Linear/{}_tensorboard'.format(os.path.dirname(opt.ckpt).split('/')[-1])

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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

    if opt.dataset == 'cifar10' or opt.dataset == 'stl10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))


    return opt


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
             dropout_rate=opt.dropout_rate,
             contrastive=False,
             timm=True,)
    else:
        model  =  SupConResNet(name=opt.model)

    if opt.coeff_smooth == 0:
        criterion1 = torch.nn.CrossEntropyLoss()
    else:
        # Implements label smoothing for CE loss
        criterion1 = LabelSmoothingCrossEntropy(opt.coeff_smooth)

    if opt.train_type == 'both':
        criterion2 = SupConLoss(temperature=opt.temp)
    else:
        criterion2 = None

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['state_dict'] if opt.model=='vit' else ckpt['model']
    # if opt.model=='resnet34':
    #     copy_keys = list(state_dict.keys())
    #     for n in copy_keys.copy():
    #         if 'module.' in n:
    #             new_key = n.split('.',2)[1]+'.'+ n.split('.',2)[0]+'.'+n.split('.',2)[-1]
    #             state_dict[new_key] = state_dict[n]
    if opt.ckpt_cl is not None:
        ckpt_cl = torch.load(opt.ckpt_cl, map_location='cpu')
        cl_state_dict = ckpt_cl['model']
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion1 = criterion1.cuda()
        if criterion2 is not None:
            criterion2 = criterion2.cuda()
        if opt.half:
            model.half()  # convert to half precision
            classifier.half()
            for layer in model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()
        cudnn.benchmark = True

        print(model.load_state_dict(state_dict,strict=False))
        #classifier.load_state_dict(cl_state_dict)

    return model, classifier, criterion1, criterion2


def train(train_loader, model, classifier, criterion1, criterion2, optimizer, epoch, opt,val_supcon=False):
    """set params for training"""
    # if opt.train_model:
    #     model.train()
    # else:
    #     model.eval()
    #
    # classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if opt.train_type == 'both':
            # concatenating images along dim 0 for contrastive loss
            images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        if not val_supcon:
            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute half precision for image
        if opt.half:
            images=images.half()

        # compute loss
        with torch.set_grad_enabled(not opt.train_model):
            if opt.model=='vit': #tod unifies these condintion to one
                features = model(images, return_emb=True)
            else:
                features = model.encoder(images )
        with torch.set_grad_enabled(True):
            output = classifier(features)

        if opt.train_type == 'both':
            # calculating contrastive loss
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt.method == 'SupCon':
                loss_con = criterion2(features, labels)
            elif opt.method == 'SimCLR':
                loss_con = criterion2(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

            # combined loss
            loss = loss_con + opt.lambda_sup * criterion1(output, labels)
        else:
            loss = criterion1(output, labels)

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
        if (idx + 1) % opt.print_freq == 0 and not val_supcon and not opt.test_contrastive_acc:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
    if val_supcon:
        return classifier
    else:
        return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion1, opt, in_out_idx=None, supcon_val=False):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top_ood = AverageMeter()
    in_conf = AverageMeter()
    out_conf = AverageMeter()

    with torch.no_grad():
        end = time.time()
        max_len = len(val_loader.dataset)
        batch_size = val_loader.batch_size
        resnet_emb = []
        for idx, (images, labels) in enumerate(val_loader):
            bsz = labels.shape[0]

            #sanity check
            if opt.ood:
                index_list = list(range(idx * batch_size, min(((idx * batch_size) + batch_size),max_len)))
                target_array = np.asarray(val_loader.dataset.targets)
                assert  (target_array[index_list] == labels.data.numpy()).all()

            images = images.float().cuda()
            labels = labels.cuda()

            if opt.half:
                images = images.half()
            # forward
            final_emb = model(images,return_feat=True) if model._get_name()=="VisionTransformer" else model.encoder(images)
            output = model.classifier(final_emb) if model._get_name()=="VisionTransformer" else classifier(final_emb)

            loss = criterion1(output, labels)

            if len(resnet_emb) == 0:
                    resnet_emb = final_emb
            else:
                    resnet_emb = torch.cat((resnet_emb, final_emb))

            #fixme for acc exclude ood, as no label is there
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

            if idx % opt.print_freq == 0 and not supcon_val and not opt.test_contrastive_acc:
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
                   #print('In dist conf:',in_score,'\t Out Dist Score:',out_score)
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                           idx, len(val_loader), batch_time=batch_time,
                           loss=losses, top1=top1))
    if not supcon_val and not opt.test_contrastive_acc:  #fixme qdd somthing loike versobe for not print
        if opt.ood:
            print(' * Acc@1 {top1.avg:.3f}\t in acc {i_acc.avg:.3f}\t in score {in_score.avg:.3f}\t out score {out_score.avg:.3f}'.format(top1=top1,i_acc=top_ood,in_score=in_conf,out_score=out_conf))
        else:
            print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    #print('Avg : In dist:',in_conf.avg,'\t Out Dist:',out_conf.avg,'\t In dist: ',top1.avg)
    return losses.avg, top1.avg, resnet_emb


def main(opt, best_accuracy=None):

    if opt.cl_save_folder is None:
        opt.cl_save_folder = os.path.dirname(opt.ckpt)
        print('data will be saved in : ',opt.cl_save_folder)
    opt.tb_folder = opt.cl_save_folder  # os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    opt.data_dir = opt.data_folder
    opt.contrastive = False

    cur_best_acc = 0

    # build data loader
    if opt.ood:
        train_loader, val_loader, out_index = set_ood_loader(opt, out_two_demo=opt.ood_two_cls_vis,
                                                             image_size=opt.image_size)
    elif opt.train_type=="both":
        train_loader, val_loader = custom_set_loader(opt)
        out_index = None
    else:
        #train_loader, val_loader = set_loader(opt,opt.image_size)
        opt.exp_name=opt.model
        opt.checkpoint_path = opt.ckpt
        train_loader, val_loader = create_dataloaders(opt)
        out_index = None
    # build model and criterion
    model, classifier, criterion1, criterion2 = set_model(opt)


    fc_params = []
    non_fc_params = []
    n_fc_params = []
    n_non_fc_params = []
        #jointly take both model and classifier param
    if 0.0 <opt.reduce_lr <1.0:
        for n, param in (list(classifier.named_parameters()) + list(model.named_parameters())):
            if param.requires_grad:
                    if n.startswith('encoder') :
                        fc_params.append(param)
                        n_fc_params.append(n)
                    else:
                        non_fc_params.append(param)
                        n_non_fc_params.append(n)
        params = [{'params': fc_params, 'lr': opt.learning_rate*opt.reduce_lr}, {'params': non_fc_params}]
        print('Reduced lr : ', n_fc_params)
        print('Not reduced lr : ', n_non_fc_params)

    else:
        params = [p for n, p in list(classifier.named_parameters())]
        if opt.train_model:
            params += [p for n, p in list(model.named_parameters()) if p.requires_grad]

    # build optimizer
    optimizer = set_optimizer(opt, params)

    # add writer
    writer =  SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)#None if opt.test_contrastive_acc else

    if not opt.use_subset:
        if not opt.dataset == 'stl10':
            tgt_idx = np.asarray(val_loader.dataset.targets)
        else:
            tgt_idx = np.asarray(val_loader.dataset.labels)
        meta_lbl = np.asarray(val_loader.dataset.classes)[tgt_idx]

    # training routine
    for epoch in range(1, opt.epochs + 1):


        adjust_learning_rate(opt, optimizer, epoch)

        if epoch % opt.save_t_SNE != 0:
            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, criterion1, criterion2,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

        # eval for one epoch
        val_loss, val_acc, resnet_emb = validate(val_loader, model, classifier, criterion1, opt, out_index)
        if epoch % opt.save_t_SNE != 0:
            print('Train epoch {}, total time {:.2f}, train accuracy:{:.2f}, val acc :{:.2f}'.format(
                epoch, time2 - time1, acc,val_acc))

        if epoch % opt.save_t_SNE == 0:  # or epoch ==1:
            if opt.ood :
                if opt.ood_two_cls_vis:
                    out_near_idx = np.where(out_index==1)[0]
                    out_far_idx = np.where(out_index == 2)[0]
                    meta_lbl[out_near_idx] = 'out_near'
                    meta_lbl[out_far_idx] = 'out_far'
                else :
                    out_lbl_idx = np.where(out_index)[0]
                    meta_lbl[:] = 'in'
                    meta_lbl[out_lbl_idx] = 'out'
            writer.add_embedding(resnet_emb, metadata=meta_lbl)
            writer.close()
            print('Saving train embedding at epoch ', epoch, ' in : ',opt.tb_folder)
            sys.exit()

        # to check for clustering in ood embedding space
        if opt.check_ood_cluster and epoch == 1:
            opt.dataset = 'cifar100'
            _, val_out_loader = set_loader(opt, opt.image_size)
            out_index = None
            _, _, resnet_ood_emb = validate(val_loader, model, classifier, criterion1, opt, out_index)
            tgt_idx = np.asarray(val_out_loader.dataset.targets)
            meta_lbl = np.asarray(val_out_loader.dataset.classes)[tgt_idx]
            writer.add_embedding(resnet_ood_emb, metadata=meta_lbl)
            writer.close()
            print('Saving ood embedding for', opt.dataset, 'at epoch ', epoch, ' in : ', opt.tb_folder)
            sys.exit()


        if val_acc > cur_best_acc:
            cur_best_acc = val_acc
            best_epoch = epoch

        if cur_best_acc > best_accuracy:
            save_file = os.path.join(opt.cl_save_folder, 'ckpt_epoch_best.pth')
            save_model(model, optimizer, opt, epoch, save_file)

        if not opt.test_contrastive_acc:
            writer.add_scalar('linear_train_loss', loss, epoch)
            writer.add_scalar('linear_val_loss', loss, epoch)
            writer.add_scalar('linear_val_acc', val_acc, epoch)
            writer.add_scalar('linear_val_loss', val_loss, epoch)
            # save features after every 100 epoch

            if epoch % opt.save_freq == 0:
                # saving re-trained model activations
                save_file = os.path.join(
                    opt.cl_save_folder, 'ckpt_retrain_epoch_{epoch}.pth'.format(epoch=epoch))
                print(save_file)
                save_model(model, optimizer, opt, epoch, save_file)
                # fixed saving classifier model
                save_file = os.path.join(
                    opt.cl_save_folder, 'ckpt_cl_epoch_{epoch}.pth'.format(epoch=epoch))
                print(save_file)
                save_model(classifier, optimizer, opt, epoch, save_file)



    print('Best accuracy: {:.2f}'.format(cur_best_acc), ' at ', best_epoch, ' for ', opt.ckpt)

    if not opt.test_contrastive_acc:
        writer.close()

    return cur_best_acc

if __name__ == '__main__':

    opt = parse_option()  #parse arguments

    if opt.test_contrastive_acc:
        checkpoints = [
            '/nfs/data3/koner/contrastive_ood/save/vit/vit_224SupCE_cifar10_bs512_lr0.01_wd1e-05_temp_0.1_210316_122535/checkpoints',
            #"/data/p.sinhamahapatra/PycharmProjects/contrastive_ood/save/SimCLR/cifar100_models/SimCLR_cifar100_resnet50_lr_0.5_dist_Cosine_decay_0.0001_bsz_512_temp_0.5_trial_11_cosine_warm/",
            #'/data/p.sinhamahapatra/PycharmProjects/contrastive_ood/save/SimCLR/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_dist_Cosine_decay_0.0001_bsz_2048_temp_0.5_trial_11_cosine_warm/'
          ]

        opt.epochs = 10
        for ckpt in checkpoints:
            val_acc_list = []
            save_epoch_list = []
            best_accuracy = 0
            best_ckpt = None
            for root, dirs, files in os.walk(ckpt):
                for file in files:
                    if file.endswith('.pth'):  # for each checkpoint file.startswith('ckpt_epoch') and
                        opt.ckpt = os.path.join(ckpt,file)
                        print("For checkpoint : ",opt.ckpt)
                        cur_best_acc = main(opt, best_accuracy)
                        val_acc_list.append(round(cur_best_acc.item(), 4))
                        save_epoch_list.append(opt.ckpt.split('_')[-1].split('.')[-2])
                        if cur_best_acc > best_accuracy:
                            best_accuracy = cur_best_acc
                            best_ckpt = opt.ckpt
            print(val_acc_list)
            print(save_epoch_list)
            pickle_obj = {'Test_acc': val_acc_list, 'Epoch': save_epoch_list}
            write_pickle(pickle_obj, os.path.join(os.path.dirname(ckpt), 'accuracy.pickle'))
            best_accuracy = round(best_accuracy.item(), 4)
            best_ckpt_dict = {'best_accuracy': best_accuracy, 'best_ckpt': best_ckpt}
            write_json(best_ckpt_dict, os.path.join(os.path.dirname(ckpt), 'best_ckpt.json'))
            print('best_accuracy:', best_accuracy, 'best_ckpt:', best_ckpt)


    else:
        best_accuracy = main(opt)


