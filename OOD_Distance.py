from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
from networks.resnet_big import SupConResNet, SupCEResNet, LinearClassifier
import os
from sklearn.metrics.pairwise import cosine_similarity
from ood_util import get_roc_sklearn
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import PCA
from torch.nn import functional as F
from vit.src.data_loaders import create_dataloaders
from vit.src.model import VisionTransformer as ViT
from vit.src.config import *
from vit.src.utils import accuracy
from evaluate_cluster_metrics import pcc_separation_ood, _unpack, global_separation
from eval_metric import create_dfs, calculate_auroc
from scipy.spatial.distance import wminkowski
import pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
def parse_option():

    parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', default='cifar10', required=False, help='cifar10 | cifar100 | stl10 | ImageNet30')
    parser.add_argument('--out_dataset', default='cifar100', required=False, help='cifar10 | cifar100 | svhn')
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--out-data-dir", type=str, default='./data', help='data folder') #specially for im30
    parser.add_argument('--outf', default='./output_maha/', help='folder to output results')
    parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
    parser.add_argument('--model', default='wide_resnet50_3', choices=['resnet18','resnet34','resnet50', 'resnet101','resnet200', 'wide_resnet50_3','vit'],
                        help='resnet50')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--ckpt', type=str, default=None, help='path to pre-trained model')
    parser.add_argument('--ood_select', type=bool, default=False,
                        help='True when particular OOD classes are included in OOD data')
    parser.add_argument('--pred_indist_lbl', type=bool, default=False, help='True when labels for in distribution samples are predicted based on argmin')
    parser.add_argument('--in_dtype', type=str, default='test', help='consider in distribution train data or test data ')

    parser.add_argument('--use_pca',type=bool, default=False, help='perform PCA reduction of embeddings')
    parser.add_argument('--use_manifold', type=str, default='None', choices=['None','TSNE','MDS'], help='perform manifold reduction of embeddings')

    parser.add_argument('--n_component', type=int, default=10, help='number of component axis')
    parser.add_argument('--param1', type=int, default=10, help='number of component axis')
    parser.add_argument('--param2', type=int, default=10, help='number of component axis')
    parser.add_argument('--param3', type=int, default=10, help='number of component axis')


    parser.add_argument('--distance', default='Euclidean', help='Euclidean | Mahalanobis | Cosine')
    parser.add_argument('--reg_factor',type=float, default=1.0, help='perform regularization on distance embedding')

    # vit related settings
    parser.add_argument("--model_arch", type=str, default="b16", help='model setting to use',
                        choices=['t16','s16','b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--image-size", type=int, default=32, help="input image size", choices=[32, 96, 128,160,224, 384])
    parser.add_argument('--compute_acc', action='store_true', help='evaluate distance based accuracy')
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    # other setting
    parser.add_argument('--contrastive', action='store_true',
                        help='using distributed loss calculations across multiple GPUs')
    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')

    opt = parser.parse_args()
    if opt.model == 'vit':
        # model config
        opt = eval("get_{}_config".format(opt.model_arch))(opt)

    # create and save out file path
    # opt.outf = opt.ckpt + '/' + opt.model + '_' + opt.dataset + '/'
    # if not os.path.isdir(opt.outf):
    #     os.makedirs(opt.outf)

    return opt

def run_model(model, loader,projection, softmax=False):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    for images, target in loader:
        total += images.size(0)
        images = images.cuda()
        if softmax:
            output, classifier = model(images, feat_cls=True)
            cls_list.append(F.softmax(classifier).data.cpu())
        else:
            output = model(images,return_feat=True if projection == "wdout_projection" else False)

        out_list.append(output.data.cpu())
        tgt_list.append(target)

    return  torch.cat(out_list), torch.cat(tgt_list), torch.cat(cls_list)

def norm_euclidean_dist(x, support_mean, reg_factor=1):
    x_dist = []
    for sample in x:
        sample_dist = []
        for support_class in support_mean:
            sample_dist.append(wminkowski(sample, support_class, p=2, w=1/reg_factor))
        x_dist.append(sample_dist)
    return torch.FloatTensor(x_dist)


def euclidean_dist(x, support_mean):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = support_mean.size(0)
    d = x.size(1)
    if d != support_mean.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    support_mean = support_mean.unsqueeze(0).expand(n, m, d)

    #return torch.pow(x - support_mean, 2).sum(2)
    return ((x - support_mean)*(x-support_mean)).sum(2)

def mahalanobis(x, support_mean, inv_covmat):
    # create function to calculate Mahalanobis distance
    n = x.size(0)
    d = x.size(1)

    #expand to fix th input size
    # x = x.unsqueeze(0).expand(m, n, d)
    # support_mean = support_mean.unsqueeze(0).expand(n, m, d)

    maha_dists = []
    for class_inv_cov, support_class in zip(inv_covmat,support_mean):
        x_mu = x - support_class.unsqueeze(0).expand(n,d)

        left = np.dot(x_mu, class_inv_cov)
        mahal = np.dot(left, x_mu.T).diagonal()
        maha_dists.append(mahal)
    return torch.from_numpy(np.asarray(maha_dists).T)

def cosine(x, support_mean, reg_factor=1):
    #calculate pairwise cosine similarity
    cosine_sim= []
    if reg_factor==1.0:
        for support_class in support_mean:
            cosine_sim.append(cosine_similarity(x, support_class.unsqueeze(0)))
        return torch.from_numpy(np.asarray(cosine_sim).squeeze(axis=2).T)
    else:
        for sample in x:
            sample_dist = []
            for support_class in support_mean:
                sample_dist.append(np.dot(sample, support_class) / (reg_factor*((np.linalg.norm(sample) * np.linalg.norm(support_class)))))
            cosine_sim.append(sample_dist)
            #np.dot(x[0], support_class) / ((np.linalg.norm(x[0]) * np.linalg.norm(support_class)))
            #cosine_sim.append(reg_factor* cosine_similarity(x, support_class.unsqueeze(0)))
        return torch.FloatTensor(cosine_sim)
            #return torch.from_numpy(np.asarray(cosine_sim).squeeze(axis=2).T)

def get_distances(train_list, test_list, out_list, classes_mean, distance, classes_feats, reg_factor =1):

    if distance=='Euclidean':
        print('Compute euclidean distance for in and out distribution data')
        if reg_factor==1.0:
            train_dists = euclidean_dist(train_list, classes_mean)
            test_dists = euclidean_dist(test_list, classes_mean)
            out_dists = euclidean_dist(out_list, classes_mean)
            dist_mean = euclidean_dist(classes_mean, classes_mean)
        else :
            train_dists = norm_euclidean_dist(train_list, classes_mean, )  # reg_factor=reg_factor
            test_dists = norm_euclidean_dist(test_list, classes_mean, reg_factor=reg_factor)  #
            out_dists = norm_euclidean_dist(out_list, classes_mean, reg_factor=reg_factor)  #
            dist_mean = norm_euclidean_dist(classes_mean, classes_mean)

    if distance =="Mahalanobis":
        print("Calculating inv covariance for support or training set")
        sup_inv_cov = [np.linalg.inv(np.cov(cls_feats, rowvar=False)) for cls_feats in classes_feats]
        print('Compute Mahalanobis distance for in and out distribution data')
        train_dists = mahalanobis(train_list, classes_mean, sup_inv_cov)
        test_dists = mahalanobis(test_list, classes_mean, sup_inv_cov)
        out_dists = mahalanobis(out_list, classes_mean, sup_inv_cov)
        dist_mean = mahalanobis(classes_mean, classes_mean, sup_inv_cov)

    if distance== "Cosine":
        print('Compute Cosine distance for in and out distribution data')
        train_dists = cosine(train_list, classes_mean, )#reg_factor=reg_factor
        test_dists = cosine(test_list, classes_mean, reg_factor=reg_factor)#
        out_dists = cosine(out_list, classes_mean, reg_factor=reg_factor) #,
        dist_mean = cosine(classes_mean, classes_mean)

    return train_dists, test_dists, out_dists, dist_mean

def main(opt, cache=None, eval_over_time_pkl=None):
    torch.cuda.manual_seed(500) # todo make seed as param
    torch.cuda.empty_cache()

    # load ckpt
    ckpt = torch.load(opt.ckpt, map_location=torch.device("cpu"))
    # converting to encoder
    if opt.model!="vit":
        for i in list(ckpt['model'].keys()):
            if i.startswith(('encoder.module')):
                parts = i.split('.', 2)
                new_key = parts[0] + '.' + parts[2]
                ckpt['model'][new_key] = ckpt['model'].pop(i)
            elif i.startswith(('module')):
                parts = i.split('.', 1)
                new_key = parts[1]
                ckpt['model'][new_key] = ckpt['model'].pop(i)

    # load networks
    if opt.model=='vit':
        model = ViT(image_size=(opt.image_size, opt.image_size),
             patch_size=(opt.patch_size, opt.patch_size),
             emb_dim=opt.emb_dim,
             mlp_dim=opt.mlp_dim,
             num_heads=opt.num_heads,
             num_layers=opt.num_layers,
             num_classes=opt.num_classes,
             attn_dropout_rate=opt.attn_dropout_rate,
             dropout_rate=opt.dropout_rate,
             contrastive=not 'supce' in opt.ckpt.lower(),
             timm=True)
    elif 'SupCE' in opt.ckpt:
        model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)
    else:
        model = SupConResNet(name=opt.model)
    model.load_state_dict(ckpt['state_dict'] if opt.model=="vit" else ckpt['model'], strict=False)
    model = model.cuda()

    print('load model: ' + opt.model)
    # Initializing PCA object if pca transform used
    if opt.use_pca:
        pca = PCA(n_components=opt.n_component)

    # load ID dataset
    print('load target data: ', opt.dataset)
    train_loader, test_loader = create_dataloaders(opt,no_train_aug=True)
    train_classes_name = np.asarray(train_loader.dataset.classes)
    test_classes_name = np.asarray(train_loader.dataset.classes)
    assert (train_classes_name == test_classes_name).all()
    # set information about feature extraction
    model.eval()

    if cache is  None:
        cache = {}
        print('Compute sample mean for training data....')
        train_emb, train_targets, train_sfmx = run_model(model,train_loader, opt.projection, 'supce' in opt.ckpt.lower())
        train_acc = float(torch.sum(torch.argmax(train_sfmx, dim=1) == train_targets)) / len(train_sfmx)
        print("Accuracy of train instances : ",train_acc)
        train_cache = []
        train_cache.append(train_emb)
        train_cache.append(train_targets)
        train_cache.append(train_sfmx)
        cache['train'] = train_cache
        test_emb, test_targets, test_sfmx = run_model(model, test_loader, opt.projection,'supce' in opt.ckpt.lower())
        test_acc = float(torch.sum(torch.argmax(test_sfmx, dim=1)==test_targets))/len(test_sfmx)
        print("Accuracy of test instances : ",test_acc)
        test_cache = []
        test_cache.append(test_emb)
        test_cache.append(test_targets)
        test_cache.append(test_sfmx)
        cache['test'] = test_cache
    else:
        print('Fetching sample mean from cache data.....')
        train_emb = cache['train'][0]
        train_targets = cache['train'][1]
        train_sfmx = cache['train'][2]
        test_emb = cache['test'][0]
        test_targets  = cache['test'][1]
        test_sfmx  = cache['test'][2]

    if opt.use_pca: # compute pca based on component features
        train_emb = torch.from_numpy(pca.fit(train_emb).transform(train_emb))
        test_emb = torch.from_numpy(pca.fit(test_emb).transform(test_emb))

    in_classes = torch.unique(train_targets)
    class_idx = [torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1) for cls in in_classes]
    classes_feats = [train_emb[idx] for idx in class_idx]
    """
    Alternate way for calculating cosine mean
    if opt.distance == 'cosine':
         # sum all normalised sample features in a class: sum_class_norm = sum (x_i / len(x_i))
         # class mean equals sum of all normalised sample features again normalised to unit length: classes_mean = sum_class_norm/len(sum_class_norm)
         sum_class_norm = torch.stack([torch.sum(torch.div(cls_feats, torch.norm(cls_feats, 2, dim=1).reshape((-1, 1))), dim=0) for cls_feats in classes_feats],dim=0)
         classes_mean = torch.stack([torch.div(cls_norms, torch.norm(cls_norms)) for cls_norms in sum_class_norm], dim=0)
     else:
    """
    classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats],dim=0)

    print('load OOD data: ', opt.out_dataset)
    _, out_test_loader = create_dataloaders(opt,no_train_aug=True, out_dataset=True)
    if not opt.out_dataset == 'SVHN':
        out_classes_name = np.asarray(out_test_loader.dataset.classes)
    else:
        out_classes_name = np.asarray(out_test_loader.dataset.labels)

    # now calculate out dist features and their distance
    out_list, out_targets, out_sfmx = run_model(model, out_test_loader, opt.projection,'supce' in opt.ckpt.lower())
    out_classes = torch.unique(out_targets)
    if opt.use_pca:
        out_list = torch.from_numpy(pca.fit(out_list).transform(out_list))

    eval_files = []

    ### iterating over both train and test ID data along with above OOD dataset
    if opt.distance=="Softmax":
        train_dist=train_sfmx; test_dist=test_sfmx; out_dists=out_sfmx;
    else:
        train_dist, test_dist, out_dists, dist_mean = get_distances(train_emb, test_emb, out_list, classes_mean, opt.distance, classes_feats,
                                                   opt.reg_factor)

    in_dtypes = ['train', 'test']
    for in_dtype in in_dtypes:
        opt.in_dtype = in_dtype
        if opt.in_dtype=='train': # assign distribution train or test
            id_dist = train_dist;id_labels = train_targets;
        else:
            id_dist = test_dist; id_labels = test_targets;


        # calculating out distribution scores which are closest to ID
        ood_lbl = torch.argmax(out_dists, dim=1) if (opt.distance == "Cosine" or opt.distance =='Softmax') else torch.argmin(out_dists, dim=1)
        ood_score = [dist[ood_lbl[i]] for i, dist in enumerate(out_dists)]

        # check whether to use original labels for test in-dist data or the predicted labels
        if opt.pred_indist_lbl:
            # predicting labels based on distance
            in_dist_lbl = torch.argmax(id_dist, dim=1) if(opt.distance == "Cosine" or opt.distance =='Softmax') else torch.argmin(id_dist, dim=1)
            in_score = [dist[in_dist_lbl[i]] for i, dist in enumerate(id_dist)]
        else:
            # using known labels for test in-dist data
            in_score = [dist[id_labels[i]] for i, dist in enumerate(id_dist)]

        # check the accuracy based on distances
        if opt.compute_acc:
            pred_lbl = torch.argsort(id_dist, dim=1, descending=True) if (opt.distance == "Cosine" or opt.distance =='Softmax') else torch.argsort(
                id_dist, dim=1)
            match_count = torch.sum((pred_lbl[:, 0] == id_labels)).data.cpu().numpy()
            print("Positive count : ", match_count)
            print("Accuracy of ", opt.in_dtype, 'are ',
                  torch.sum((pred_lbl[:, 0] == id_labels)).data.cpu().numpy() / len(id_labels))
            in_assigned_class = [torch.where(id_labels == in_tgt_lbl) for in_tgt_lbl in in_classes]
            in_assigned_class_score = [j[i] for i, j in zip(id_labels, id_dist)]
            in_assigned_class_score = [np.array(in_assigned_class_score)[i[0]] for i in
                                       in_assigned_class]  # class wise score
            out_assigned_class = [torch.where(ood_lbl == in_tgt_lbl)[0] for in_tgt_lbl in
                                  in_classes]  # all out assigned classes based on distance
            out_assigned_class_score = [np.array(ood_score)[i] for i in out_assigned_class]
            print("Avg out class score : ", [np.mean(i) for i in out_assigned_class_score])
            print("Avg in class score : ", [np.mean(i) for i in in_assigned_class_score])
        # return train_cache

        # now stack score and lbl both for in and out
        # save format (distance socre, pred cls lbl, true cls lbl,  in/out dist len)
        if opt.pred_indist_lbl:
            in_data_full = np.stack(
                (in_score, train_classes_name[in_dist_lbl], train_classes_name[id_labels], np.ones(len(in_score))),
                axis=1)
        else:
            in_data_full = np.stack(
                (in_score, train_classes_name[id_labels], train_classes_name[id_labels], np.ones(len(in_score))),
                axis=1)

        out_data_full = np.stack((ood_score, train_classes_name[ood_lbl], out_classes_name[out_targets],
                                  np.zeros(len(ood_score))), axis=1)
        in_out_data = np.concatenate((in_data_full, out_data_full))

        if in_dtype=='test':
            if opt.distance == "Cosine" or opt.distance =='Softmax':
                auroc_ssd = get_roc_sklearn(ood_score, in_score)
            else:
                auroc_ssd = get_roc_sklearn(in_score,ood_score)
            print("SSD AUROC {0} and Distance {1}".format(auroc_ssd,opt.distance))

        if opt.pred_indist_lbl:
            file_name = os.path.join(os.path.dirname(opt.ckpt), '%s_%s_%s_predIN_%s.npy' % (
            opt.distance, opt.out_dataset, opt.ckpt.split('_')[-1].split('.')[-2], opt.in_dtype))
        else:
            file_name = os.path.join(os.path.dirname(opt.ckpt), '%s_%s_%s_GT_%s.npy' % (
            opt.distance, opt.out_dataset, opt.ckpt.split('_')[-1].split('.')[-2], opt.in_dtype))
        print('Now concat both in and out distribution and save in ', file_name)
        np.save(file_name, in_out_data)

        eval_files.append(file_name)

        """
        # Calculating cluster quality evaluation metrics
        
        if opt.in_dtype == 'test':
            print('\nBased on', opt.distance, 'distance \n')
            if opt.distance == 'Cosine':
                dist_metric = 'cosine'
            elif opt.distance == 'Euclidean':
                dist_metric = 'euclidean'

            if opt.pred_indist_lbl:
                target_label = in_dist_lbl
                target_type = 'predicted test samples'
            else:
                target_label = in_targets
                target_type = 'GT test samples'

            #     ## calculating class-wise distances for separation (pcc) between in-distribution and ood samples
            #     print('PCC:')
            #     target_in_ood_list = np.concatenate((in_list, out_list), axis=0)
            #     target_in_ood_labels = np.concatenate((in_targets, (out_targets + len(in_classes))), axis=0)
            #     pcc = pcc_separation_ood(target_in_ood_list, target_in_ood_labels, metric=dist_metric,
            #                              num_class=len(in_classes))
            #     save_file = np.asarray(pcc)
            #     file_name = os.path.join(os.path.dirname(opt.ckpt), 'Pcc-classwise_%s_dist_%s_ood_%s.npy' % (
            #         dist_metric, opt.epoch, opt.out_dataset))
            #     np.save(file_name, save_file)
            #
            #     print('Global Separation:')
            #     num_samples = 9
            #     #tr = np.linspace(0.2, 1, num_samples)
            #     gsc = global_separation(in_list, target_label, metric=dist_metric, num_samples=num_samples)
            #     print(list(gsc))
            #     save_file = np.asarray(gsc)
            #     file_name = os.path.join(os.path.dirname(opt.ckpt), 'GSC-classwise_%s_dist_%s.npy' % (dist_metric, opt.epoch))
            #     np.save(file_name, save_file)
            #
            #
            # ## calculating silhouette coefficient on in-distribution test labels
            # print("Silhouette Coefficient for %s : %0.3f" % (target_type, silhouette_score(in_list, target_label, metric=dist_metric,
            #                                                          sample_size=in_data)))
            # sample_silhouette_values = silhouette_samples(in_list, target_label)
            # means_list = []
            # median_list = []
            # for label in range(len(test_classes_name)):
            #     s = sample_silhouette_values[target_label == label]
            #     means_list.append(s.mean())
            #     median_list.append(np.median(s))
            # print('\n', 'Class Mean list:', means_list)
            # print(list(zip(test_classes_name, means_list)))
            # print('\n', 'Class Median list:', median_list)
            #
            # ## calculating NMI between Gt and predicted class labels
            # if opt.pred_indist_lbl:
            #     print('NMI between GT and predicted test samples :', normalized_mutual_info_score(in_targets, in_dist_lbl))
            
        """

    ### Evaluation Metrics

    dfs = create_dfs(eval_files)
    auroc, tnr_at_tpr95, aupr = calculate_auroc(dfs, opt.distance, opt.dataset)

    # creating a pickle dump of evaluation metrics across time steps
    if eval_over_time_pkl is None:
        OOD_data = [] ; Epoch = []; Distance = []; GSC = []; AUROC = [];
    else:
        OOD_data = eval_over_time_pkl['OOD']; Epoch = eval_over_time_pkl['Epoch']; Distance = eval_over_time_pkl['Distance'];
        GSC = eval_over_time_pkl['GSC']; AUROC = eval_over_time_pkl['AUROC']

    OOD_data.append(opt.out_dataset); Epoch.append(opt.epoch); Distance.append(opt.distance); AUROC.append(auroc)
    num_samples_gsc = 1 # global separation corresponding to 100% of test samples
    if opt.distance=="Softmax":
        GSC.append(0)
    else:
        GSC.append(_unpack(global_separation(test_dist, test_targets, metric=opt.distance, num_samples=num_samples_gsc)))

    eval_over_time_pkl = {'OOD': OOD_data, 'Epoch': Epoch, 'Distance': Distance, 'GSC': GSC, 'AUROC': AUROC}


    with open(os.path.join(os.path.dirname(opt.ckpt), "auroc.txt"), "a") as output:
        if 'train_acc' in locals():
            output.write(
                "Training Accuracy : {0} :: Test Accuracy : {1} .\n".format(train_acc, test_acc ))
        if opt.use_pca:
            print("AUROC_PCA : {0} :: epoch : {4} :: distance  {1}  :: in : {2} :: out : {3} :: axis {5} :: axis {6}.\n".format(
                    round(auroc, 4), opt.distance, opt.dataset, opt.out_dataset, opt.ckpt.split('_')[-1].split('.')[-2],
                    opt.n_component, opt.projection))
            output.write("AUROC_PCA : {0} :: epoch : {4} :: distance  {1}  :: in : {2} :: out : {3} :: axis {5} :: axis {6}.\n".format(
                round(auroc, 4), opt.distance, opt.dataset, opt.out_dataset, opt.ckpt.split('_')[-1].split('.')[-2], opt.n_component,  opt.projection))
        else:
            print("AUROC : {0} :: AUROC SSD : {7} :: epoch : {4} :: distance  {1}  :: in : {2} :: out : {3} :: axis {5} :: reg_factor : {6}.\n".format(
                round(auroc, 4), opt.distance, opt.dataset, opt.out_dataset, opt.ckpt.split('_')[-1].split('.')[-2], opt.projection, opt.reg_factor, auroc_ssd))
            output.write("AUROC : {0} :: AUROC SSD : {6} :: epoch : {4} :: distance  {1}  :: in : {2} :: out : {3} :: axis {5}.\n".format(
                round(auroc, 4), opt.distance, opt.dataset, opt.out_dataset, opt.ckpt.split('_')[-1].split('.')[-2], opt.projection, auroc_ssd))

        output.close()
    return cache, eval_over_time_pkl


def model_arch(opt, ckpt_path):
    if 'resnet34' in ckpt_path: # select model
        opt.model = 'resnet34';
    elif 'resnet50' in ckpt_path:
        opt.model = 'resnet50'
    elif 'resnet101' in ckpt_path:
        opt.model = 'resnet101'
    elif 'vit' in ckpt_path.lower() or 'deit' in ckpt_path.lower():
        opt.image_size = 160 if '160' in os.path.basename(ckpt_path) else 224
        opt.model = 'vit'
        if 'b32' in os.path.basename(ckpt_path):
            opt.model_arch = 'b32';opt = get_b32_config(opt)
        elif 'b16' in ckpt_path or 'vit' in ckpt_path.lower():
            opt.model_arch='b16';opt = get_b16_config(opt)
        elif 's16' in ckpt_path:
            opt.model_arch='s16'; opt = get_s16_config(opt)
        else:
            opt.model_arch='t16'; opt = get_t16_config(opt)
    if 'resnet' in ckpt_path:
        opt.image_size = 224 if opt.dataset=='ImageNet' else 32
    return opt

def run_ood_distance(opt):
    root_dir = '/nfs/data3/koner/save/vit'#specify the root dir
    root_vit = 'experiments/save'
    ckpt_list = ['vit_224SupCE_SupCE_cifar10_b16_bs256_lr0.01_wd1e-05_temp_0.1_210622_002955'
        # 'deit_224_SupCE_cifar10_s16_bs256_lr0.01_wd1e-05_temp_0.1_210531_070807',
        #         'deit_224_SupCE_cifar10_t16_bs256_lr0.01_wd1e-05_temp_0.1_210531_103317',
        #          'vit_224SupCE_cifar10_bs512_lr0.01_wd1e-05_temp_0.1_210316_122535',
        #           'vit_224SupCE_SupCE_cifar10_b16_bs256_lr0.01_wd1e-05_temp_0.1_210622_002955',
        #          'vit_224_SupCE_cifar100_bs256_lr0.01_wd1e-05_temp_0.1_210411_204950',
        #          'vit_160_alb_im30_SupCE_ImageNet_bs256_lr0.01_wd1e-05_temp_0.1_210425_005746',
                 ]
    # 'SupCE/ImageNet_models/SupCE_ImageNet_resnet101_lr_0.8_decay_0.0001_bsz_64_trial_0_seed_None_cosine',

    ckpt_path_list = [opt.ckpt] if opt.ckpt != None else ckpt_list
    distances = ['Softmax','Cosine','Euclidean']#,] , 'Mahalanobis',
    feat_vector = ['wdout_projection' ] # 'with_projection'
    im_out_data_list = ['Places365','Dtd','Dogs','Food','Caltech256', 'CUB',]
    cifar10_out_data_list = ['cifar100','Imagenet_resize', 'LSUN_resize','SVHN']
    cifar100_out_data_list = ['cifar10','Imagenet_resize', 'LSUN_resize',  'SVHN']
    use_pca = [False]
    n_component = [128]#3,10,
    reg_factor =[1.0] #[1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.5]# regularizer for distance embedding
    param1 = [5,10,15,25,35,50] # for TSNE its preplixity
    param2 = [20,50,80,100,150,200,300] # for TSNE its LR
    param3 = [100,250,200,250,300,400,500,600]
    for ckpt_path in ckpt_path_list: # set all dataset param for diff train model
        if 'ImageNet' in ckpt_path:
            opt.dataset='ImageNet';opt.num_classes =30;opt.data_dir='data/ImageNet30';out_dataset_list=im_out_data_list
        elif 'cifar100' in ckpt_path:
            opt.dataset = 'cifar100';opt.num_classes = 100;out_dataset_list=cifar100_out_data_list
        else :
            opt.dataset = 'cifar10';opt.num_classes = 10;out_dataset_list = cifar10_out_data_list

        opt = model_arch(opt,ckpt_path)

        if opt.model=='vit':
            ckpt_path = os.path.join(root_dir,ckpt_path,'checkpoints'); opt.batch_size= 256
        else:
            ckpt_path = os.path.join(root_dir, ckpt_path)

        if os.path.isfile(os.path.join(ckpt_path,'auroc.txt')): # earlier Auroc file exist then remove it
            os.remove(os.path.join(ckpt_path,'auroc.txt'))

        for root, dirs, files in os.walk(ckpt_path):
            eval_over_time_pkl = None
            for file in files:
                if file.endswith('.pth'): # for each checkpoint #file.startswith('ckpt_epoch') and
                    cache = None #reinit for every checkpoint
                    for out_dataset in out_dataset_list:
                        opt.out_dataset = out_dataset
                        for reg in reg_factor:
                            opt.reg_factor=reg
                            for pca in use_pca: # compute wd PCA and its axis
                                opt.use_pca = pca
                                if opt.use_pca:
                                    for axis in  n_component:
                                        opt.n_component = axis
                                        for distance in distances:  # for each distances
                                            # for in_dtype in in_dtypes:  # for each dtypes
                                            for projection in feat_vector:  # compute with or wdout 128 projection head
                                                opt.ckpt = os.path.join(root, file)
                                                opt.epoch = opt.ckpt.split('_')[-1].split('.')[-2]
                                                opt.distance = distance
                                                opt.projection = projection
                                                cache, eval_over_time_pkl = main(opt, cache, eval_over_time_pkl)
                                else:
                                    for distance in distances:  # for each distances
                                        # for in_dtype in in_dtypes:  # for each dtypes
                                        for projection in feat_vector:  # compute with or wdout 128 projection head
                                            opt.ckpt = os.path.join(root, file)
                                            opt.epoch = opt.ckpt.split('_')[-1].split('.')[-2]
                                            opt.distance = distance
                                            opt.projection = projection
                                            print("All params : ", opt)
                                            cache, eval_over_time_pkl = main(opt, cache, eval_over_time_pkl)
            if eval_over_time_pkl is not None:
                fname = os.path.join(os.path.dirname(ckpt_path), 'Eval-cluster-over-time.pickle')
                with open(fname, 'wb') as f1:
                    pickle.dump(eval_over_time_pkl, f1)
                            # for manifold in opt.use_manifold:
                            #     if manifold is not 'None':
                            #             for axis in n_component:
                            #                 opt.n_component = axis
                            #                 for par1 in param1:
                            #                     for distance in distances:  # for each distances
                            #                         # for in_dtype in in_dtypes:  # for each dtypes
                            #                         for projection in feat_vector:  # compute with or wdout 128 projection head
                            #                             opt.ckpt = os.path.join(root, file)
                            #                             opt.epoch = os.path.splitext(file)[0].split('_', 1)[1]
                            #                             opt.distance = distance
                            #                             opt.projection = projection
                            #                             cache = main(opt, cache)


if __name__ == '__main__':
    #parse argument
    opt = parse_option()
    run_ood_distance(opt)
