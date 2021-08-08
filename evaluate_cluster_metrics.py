from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import os
import pickle

from torchvision import transforms
from torch.autograd import Variable
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, silhouette_samples, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from itertools import chain

from networks.resnet_big import SupConResNet, SupCEResNet, LinearClassifier
from main_ce import set_loader

pca_components = [0, 10, 20, 30]#[0, 10, 20, 30, 45, 60]
num_clusters = [0, 10, 15, 18, 20, 22, 25, 30] #[0, 8, 10, 15, 20, 22, 25, 28, 30, 35, 40, 50, 60]

def parse_option():
    # fixme move default ckpt to param
    parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')

    parser.add_argument('--default_path', type=str, default= '/data/p.sinhamahapatra/PycharmProjects/contrastive_ood/save/SimCLR/cifar10_models/SimCLR_cifar10_resnet50_lr_0.5_dist_Cosine_decay_0.0001_bsz_2048_temp_0.5_trial_11_cosine_warm/',
                        help = 'path to pre-trained model')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for data loader')
    parser.add_argument('--dataset', default='cifar10', required=False, help='cifar10 | cifar100 | svhn')
    parser.add_argument('--out_dataset', default='cifar100', required=False, help='cifar10 | cifar100 | svhn')
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--outf', default='./output_maha/', help='folder to output results')
    parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
    parser.add_argument('--net_type', default='wide_resnet50_3', choices=['resnet18','resnet34','resnet50', 'resnet101','resnet200', 'wide_resnet50_3'],
                        help='resnet50')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--ckpt', type=str, default=None, help='path to checkpoints of pre-trained model')
    parser.add_argument('--ood_select', type=bool, default=False,
                        help='True when particular OOD classes are included in OOD data')
    parser.add_argument('--pred_indist_lbl', type=bool, default=False, help='True when labels for in distribution samples are predicted based on argmin')
    parser.add_argument('--in_dtype', type=str, default='test', help='consider in distribution train data or test data ')

    parser.add_argument('--distance', default='Euclidean', help='Euclidean | Mahalanobis | Cosine')
    # arguments added for subset class
    parser.add_argument('--use_subset', action='store_true', help='sub set of classes')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')


    opt = parser.parse_args()

    #opt.data_folder = './datasets/'
    """
    # create and save out file path
    opt.outf = opt.outf + opt.net_type + '_' + opt.dataset + '/'
    if not os.path.isdir(opt.outf):
        os.makedirs(opt.outf)
    """

    return opt

def run_model(model, loader):
    #run the resnet model+
    total = 0
    out_list = []
    all_targets = []
    for images, target in loader:
        total += images.size(0)
        images = images.cuda()
        output = model(images,return_feat=True)
        if len(out_list)==0:
            out_list = output.data.cpu()
            all_targets = target.data.cpu()
        else:
            out_list = torch.cat((out_list,output.data.cpu()),0)
            all_targets = torch.cat((all_targets,target.data.cpu()))

    return out_list, all_targets, total

def __local_purity(H, y, nn):
    """
    :param H: embedding to evaluate
    :param y: ground-truth classes
    :param nn: number of neighbours to consider
    """
    A = kneighbors_graph(H, nn + 1, include_self=True)
    neigbourhoods = A.dot(np.eye(y.max() + 1)[y])
    frequencies = neigbourhoods / neigbourhoods.sum(1)[:, None]
    purity = frequencies.max(axis=1)
    return purity.mean()


def local_purity(H, y, nn=None, num_samples=10):
    """
    :param H: embedding to evaluate
    :param y: ground-truth classes
    :param nn: number of neighbours to consider, if nn=None evaluate for nn=[1...size of max cluster]
    :param num_samples: number of samples in the range (1, size of max cluster)
    """
    if nn is None:
        max_size_cluster = np.unique(y, return_counts=True)[1].max()
        res = np.fromiter((__local_purity(H, y, nn)
                            for nn in np.linspace(0, max_size_cluster, num_samples).astype(np.int32)), np.float32)
        return res
    else:
        res = __local_purity(H, y, nn)
        return res

def __P_c_cp(distances, y, c, cp, tr, pcc_dist=False):
    pcc = distances[y == c][:, y == cp].flatten()

    if tr < 1.0:
        k_smallest = int(len(pcc) * tr)
        idx = np.argpartition(pcc, k_smallest)
        pcc_mean = pcc[idx[:k_smallest]].mean()
        pcc_distr = pcc[idx[:k_smallest]]
        #return pcc[idx[:k_smallest]].mean()
    else:
        pcc.sort()
        pcc_mean = pcc[:int(len(pcc) * tr)].mean()
        pcc_distr = pcc[:int(len(pcc) * tr)]
        #return pcc[:int(len(pcc) * tr)].mean()
    if not pcc_dist:
        return pcc_mean
    else:
        return pcc_distr

def __GS_c(distances, y, c, tr):
    all_c = np.arange(np.int(y.max()) + 1)
    other_c = np.setdiff1d(all_c, c)

    pcc = np.fromiter((__P_c_cp(distances, y, c, cp, tr) for cp in all_c), np.float32)
    pcc_min = pcc[other_c].min()

    return np.true_divide(pcc_min - pcc[c], np.maximum(pcc_min, pcc[c]))

def __P_c_c(distances, y, tr, c):
    all_c = np.arange(y.max() + 1)
    pcc = np.fromiter((__P_c_cp(distances, y, c, cp, tr) for cp in all_c), np.float32)
    return pcc

def __P_c_c_dist(distances, y, tr, c, pcc_dist=True):
    all_c = np.arange(y.max() + 1)
    pcc_dist = np.fromiter(chain.from_iterable(__P_c_cp(distances, y, c, cp, tr, pcc_dist=pcc_dist) for cp in all_c), np.float32)
    return pcc_dist

def global_separation(H, y, metric='cosine', num_samples=9, k=None):
    """
    :param H: embedding to evaluate
    :param y: ground-truth classes
    :param k: if None evaluate all classes else only class k
    :param num_samples: number of samples in the range (0, 100)%
    :return:
    """
    distances = squareform(pdist(H, metric))
    if k is None:
        ranged = range(y.max() + 1)
    else:
        ranged = [k]

    if num_samples != 1:
        res = np.fromiter((__GS_c(distances, y, c, tr)
                        for c in ranged
                        for tr in np.linspace(0.2, 1, num_samples)), np.float32).reshape((-1, num_samples))
    else:
        tr = 1
        res = np.fromiter((__GS_c(distances, y, c, tr)
                           for c in ranged), np.float32).reshape((-1, num_samples))
    return res

def pcc_separation_ood(H, y, metric='cosine', num_class=10, pcc_dist=True, k=None):
    """
    :param H: embedding to evaluate
    :param y: ground-truth classes
    :param k: if None evaluate all classes else only class k
    :param num_class: number of classes in the indistribution data%
    :return: Pairwise separation distances between classes
    """
    distances = squareform(pdist(H, metric))
    tr = 1 # calculating for all samples
    if k is None:
        ranged = range(num_class)
    else:
        ranged = [k]

    #res = np.fromiter((__P_c_c(distances, y, tr, c)
    #                    for c in ranged), np.float32).reshape((-1, num_class))
    res = []
    res_dist = []
    for c in ranged:
        res.append(__P_c_c(distances, y, tr, c))
        #if pcc_dist:
        #   res_dist.append(__P_c_c_dist(distances, y, tr, c))

    res = np.vstack(res)
    #res_dist = np.vstack(res_dist)

    return res#, res_dist

def _unpack(arr):
    a = list(arr)
    b = []
    for i in range(len(a)):
        b.append(*a[i])
    return b

def cluster_purity(k_labels, in_labels):
    """
    A function to evaluate cluster purity  after K-means clustering is performed.

    For a given cluster identified after K-means clustering, all the GT labels corresponding
    to the samples in this cluster are checked.
    A K-means cluster has maximum purity when the GT labels for all the samples in the cluster are same.
    Thus, cluster purity is taken as the maximum percentage of the samples having similar GT label.

    : return: list of purity percentages for each K-means cluster
    """
    # ToDo: Convert to generic numpy format

    k_classes = torch.unique(k_labels)
    k_class_idx = [torch.nonzero(torch.eq(cls, k_labels)).squeeze(dim=1) for cls in k_classes]

    in_classes_in_k_clstr = [in_labels[idx] for idx in k_class_idx]
    purity_list = []

    for cluster_k in in_classes_in_k_clstr:
        unique, counts = torch.unique(cluster_k, return_counts=True)
        purity_list.append(np.round(np.asarray(counts).max()/len(cluster_k), 5))

    return purity_list


def eval_cluster_quality(in_list, in_targets, pca_components=pca_components, kmeans_clusters=num_clusters, ckpt=None, distances=['cosine', 'euclidean'], projection=None):
    """
    A function used to evaluate cluster quality.

    The embeddings are clustered using K-means clustering for varying cluster sizes.
    Also, for higher dimensional embeddings PCA has been deployed for dimensionality reduction for varying number of axes
    and then followed by K-means clustering. All the combinations of PCA components and cluster sizes are evaluated using
    Global Separation, Cluster Purity and Local Purity metrics (optional, not yet included).
    When PCA components = 0, K clusters = 0 : Cluster quality of the embeddings of the original trained model is analysed
    Finally, an optimum number of PCA components and K-means  is to be determined from the raw data saved in pickled object.

    :return: pickled object with all variables and evaluation metrics
    """

    epoch = os.path.splitext(os.path.basename(ckpt))[0].split('_', 1)[1]

    distances = distances #['cosine', 'euclidean']


    for distance in distances:

        Distance = []
        PCA_axis = []
        K = []
        GSC = []
        ClusterPurity = []

        fname = os.path.join(os.path.dirname(ckpt), 'KCluster-all_%s_%s.pickle' % (distance, projection))
        for pca_component in pca_components:

            if pca_component > 0:
                pca = PCA(n_components=pca_component)
                new_in_list = torch.from_numpy(pca.fit(in_list).transform(in_list))
            else:
                # pca_component = 0 considers the default in_list with all the dimensions of embeddings
                new_in_list = in_list
            for num_cluster in kmeans_clusters:
                num_samples_gsc = 1
                if num_cluster > 0:
                    kmeans = KMeans(init="k-means++", n_clusters=num_cluster, n_init=20, random_state=123).fit(
                        new_in_list)
                    kmeans_in_targets = torch.from_numpy(kmeans.labels_)

                    gsc = _unpack(
                        global_separation(new_in_list, kmeans_in_targets, metric=distance, num_samples=num_samples_gsc))
                    cp = cluster_purity(kmeans_in_targets, in_targets)
                else:
                    gsc = _unpack(global_separation(new_in_list, in_targets, metric=distance, num_samples=num_samples_gsc))
                    cp = np.zeros_like(gsc)

                Distance.append(distance)
                PCA_axis.append(pca_component)
                K.append(num_cluster)
                GSC.append(gsc)
                ClusterPurity.append(cp)
        print('Cluster quality evaluation metrics for K-means clustering using {} distance saved ... '.format(distance))

        pickle_obj = {'Distance': Distance, 'PCA': PCA_axis, 'K': K, 'GSC': GSC, 'CP': ClusterPurity}

        with open(fname, 'wb') as f1:
            pickle.dump(pickle_obj, f1)



def main(opt):
    # set the path to pre-trained model and output
    # pre_trained_net = './pre_trained/' + opt.net_type + '_' + opt.dataset + '.pth
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.manual_seed(100)
    # torch.cuda.set_device(opt.gpu)
    torch.cuda.empty_cache()

    # load ckpt
    ckpt = torch.load(opt.ckpt)
    # converting to encoder
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
    if 'SupCE' in opt.ckpt:
        model = SupCEResNet(name=opt.net_type)
    else:
        model = SupConResNet(name=opt.net_type)
    model.load_state_dict(ckpt['model'])

    model = model.cuda()

    in_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    print('load model: ' + opt.net_type)

    # load dataset
    print('load target data: ', opt.dataset)

    if not opt.use_subset:
        train_loader, test_loader = data_loader.getTargetDataSet(opt.dataset, opt.batch_size, in_transform, opt.dataroot)

        train_classes_name = np.asarray(train_loader.dataset.classes)
        test_classes_name = np.asarray(train_loader.dataset.classes)
        assert (train_classes_name == test_classes_name).all()

    else:
        # using subset of classes, currently only cifar10
        train_loader, test_loader = set_loader(opt)

    print('load OOD data: ', opt.out_dataset)
    out_test_loader = data_loader.getNonTargetDataSet(opt.out_dataset, opt.batch_size, in_transform, opt.dataroot)
    if not opt.out_dataset == 'svhn':
        out_classes_name = np.asarray(out_test_loader.dataset.classes)
    else:
        out_classes_name = np.asarray(out_test_loader.dataset.labels)
    print('OOD classes: ', out_classes_name)

    # set information about feature extraction
    model.eval()

    in_list, in_targets, in_data = run_model(model, test_loader)

    # now calculate out dist features and their distance
    out_list, out_targets, out_data = run_model(model, out_test_loader)

    ### For testing cluster quality without OOD data
    eval_cluster_quality(in_list, in_targets, pca_components=pca_components, kmeans_clusters=num_clusters, ckpt=opt.ckpt)

    """
    X = in_list.numpy()
    y = in_targets.numpy()

    num_in_class = len(np.unique(y))

    print(f'Cluster quality measures for {opt.net_type} model trained for {opt.epoch} : ')
    distance_metrics = ['cosine', 'euclidean']
    for dist in distance_metrics:
        print('\nBased on', dist, 'distance \n')

        print('Pcc:')
        target_in_ood_list = np.concatenate((in_list, out_list), axis=0)
        target_in_ood_labels = np.concatenate((in_targets, (out_targets + num_in_class)), axis=0)
        pcc = pcc_separation_ood(target_in_ood_list, target_in_ood_labels, metric=dist,
                                 num_class=num_in_class)
        save_file = np.asarray(pcc)
        file_name = os.path.join(os.path.dirname(opt.ckpt), 'Pcc-classwise_%s_dist_%s_ood_%s.npy' % (
            dist, opt.epoch, opt.out_dataset))
        np.save(file_name, save_file)

        print('Global Separation:')
        num_samples = 9
        tr = np.linspace(0.2, 1, num_samples)
        gsc = global_separation(X, y, metric=dist, num_samples=num_samples)
        print(tr)
        print(list(gsc))
        save_file = np.asarray(gsc)
        file_name = os.path.join(os.path.dirname(opt.ckpt), 'GSC-classwise_%s_dist_%s.npy' % (dist, opt.epoch))
        np.save(file_name, save_file)

        print("Silhouette Coefficient: %0.3f" % silhouette_score(X, y, metric=dist, sample_size=in_data))
        sample_silhouette_values = silhouette_samples(X, y)
        means_list = []
        for label in range(num_in_class):
            means_list.append(sample_silhouette_values[y == label].mean())
        print(means_list)
        save_file = np.asarray(means_list)
        #print(list(zip(test_classes_name, means_list)))
        file_name = os.path.join(os.path.dirname(opt.ckpt), 'SC-classwise_%s_dist_%s.npy' % (dist, opt.epoch))
        np.save(file_name, save_file)

        print("Local Purity:")
        max_size_cluster = np.unique(y, return_counts=True)[1].max()
        num_cluster_sample = len(test_classes_name) + 1
        cluster_samples = np.linspace(0, max_size_cluster, num_cluster_sample)
        print('cluster sizes', cluster_samples)
        loc_purity = local_purity(X, y, nn=None, num_samples=11)
        save_file = np.asarray(loc_purity)
        file_name = os.path.join(os.path.dirname(opt.ckpt), 'Local-Purity_%s_dist_%s.npy' % (dist, opt.epoch))
        np.save(file_name, save_file)
        print(loc_purity)
        """


if __name__ == '__main__':
    #parse argument
    opt = parse_option()
    ckpt_path = opt.default_path

    for root, dirs, files in os.walk(ckpt_path):
        for file in files:
            if file.startswith('ckpt_epoch') and file.endswith('.pth'): # for each checkpoint
                opt.ckpt = os.path.join(root, file)
                opt.epoch = os.path.splitext(file)[0].split('_', 1)[1]
                main(opt)


