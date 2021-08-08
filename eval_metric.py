import os
import json
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics

cifar10_class_order = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar100_class_order = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
                        'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                        'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                        'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
                        'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                        'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                        'worm']
im30_class_order = ['acorn', 'airliner', 'ambulance', 'american_alligator', 'banjo', 'barn', 'bikini', 'digital_clock',
                    'dragonfly', 'dumbbell', 'forklift', 'goblet', 'grand_piano', 'hotdog', 'hourglass', 'manhole_cover', 'mosque', 'nail',
                    'parking_meter', 'pillow', 'revolver', 'rotary_dial_telephone', 'schooner', 'snowmobile', 'soccer_ball', 'stingray', 'strawberry', 'tank', 'toaster', 'volcano']
subset_class_order = ['airplane', 'automobile', 'bird', 'cat']

stl10_class_order = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

class_list = {'cifar10': cifar10_class_order, 'cifar100': cifar100_class_order, 'ImageNet':im30_class_order, 'stl10': stl10_class_order}#'':

def get_class_list(class_name):
    cl_list = class_list[class_name]
    return cl_list


def create_dfs(files):
    col_names_last = ['feat_dist', 'predicted_class', 'true_class', 'sample_type']

    dfs = pd.DataFrame(columns=col_names_last)
    for f in files:
        data = np.load(f)
        df = pd.DataFrame(data=data[0:, 0:], index=[i + 1 for i in range(data.shape[0])], columns=col_names_last)
        df[['feat_dist', 'sample_type']] = df[['feat_dist', 'sample_type']].apply(pd.to_numeric)

        df.loc[(df['sample_type'] != 1.0), 'sample_type'] = 'out'

        if f.split('_')[-1].split('.')[-2] == 'train':
            df.loc[(df['sample_type'] != 'out'), 'sample_type'] = 'in_train'

            dfs = dfs.append(df[df.sample_type == 'out'], ignore_index=True)
            dfs = dfs.append(df[df.sample_type == 'in_train'], ignore_index=True)

        else:
            df.loc[(df['sample_type'] != 'out'), 'sample_type'] = 'in_test'
            dfs = dfs.append(df[df.sample_type == 'in_test'], ignore_index=True)

    return dfs


def calculate_auroc(df_all, distance, dataset_in, use_clustering=False, percentile=[0.0, 1.0]):

    percentiles = np.linspace(percentile[0], percentile[1], int(100 * (percentile[1] - percentile[0])) + 1)
    stats = df_all[df_all['sample_type'] == 'in_train'].groupby(['true_class']).describe(percentiles=percentiles)

    df_ = df_all[df_all['sample_type'] != 'in_train']  # taking only validation and ood samples

    df_pr = pd.DataFrame(columns=['precision', 'recall', 'FPR', 'TPR', 'percentile'])

    if not use_clustering:
        in_class_list = get_class_list(dataset_in)
    else:
        in_class_list = df_all[df_all['sample_type'] == 'in_train'].true_class.unique()

    for p in percentiles:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        p_percent = int(p * 100)

        for cls in in_class_list:
            df_cls = df_[df_.predicted_class == cls]

            t = stats['feat_dist']['{}%'.format(p_percent)][cls]

            if not distance == 'Cosine':
                tp = tp + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist >= t)])
                fp = fp + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist >= t)])
                fn = fn + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist < t)])
                tn = tn + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist < t)])
            else:
                tp = tp + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist <= t)])
                fp = fp + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist <= t)])
                fn = fn + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist > t)])
                tn = tn + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist > t)])
            """
            if not distance == 'Cosine':
                tp = tp + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist <= t)])
                fp = fp + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist <= t)])
                tn = tn + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist > t)])
                fn = fn + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist > t)])

            else:
                tp = tp + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist >= t)])
                fp = fp + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist >= t)])
                tn = tn + len(df_cls.loc[(df_cls.sample_type == 'out') & (df_cls.feat_dist < t)])
                fn = fn + len(df_cls.loc[(df_cls.sample_type == 'in_test') & (df_cls.feat_dist < t)])
            """

        if ((tp != 0) or (fp != 0)):
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            fpr = fp / (tn + fp)

            df_pr = df_pr.append({'precision': precision, 'recall': recall, 'FPR': fpr, 'TPR': recall, 'percentile': p},
                                 ignore_index=True)

    #### AUROC calculation
    df_sort = df_pr.sort_values(by=['FPR'], ascending=False)
    #auroc = - np.trapz(df_sort['TPR'], df_sort['FPR'])

    auroc = metrics.auc(df_sort['FPR'], df_sort['TPR'])

    #### AUPR calculation

    df_sort = df_pr.sort_values(by=['recall'], ascending=False)
    # aupr = - np.trapz(df_sort['precision'], df_sort['recall'])
    aupr = metrics.auc(df_sort['recall'], df_sort['precision'])

    ### TNR@TPR95 calculation
    """
    fpr_at_tpr95 = df_pr[(round(df_pr['TPR'], 2) == 0.95)]['FPR']
    if len(fpr_at_tpr95) == 0:
        fpr_at_tpr95 = df_pr[(round(df_pr['TPR'], 2) == 0.96)]['FPR']
        if len(fpr_at_tpr95) == 0:
            fpr_at_tpr95 = df_pr[(round(df_pr['TPR'], 2) == 0.94)]['FPR']

    if not len(fpr_at_tpr95) == 0:
        tnr_at_tpr95 = 1 - fpr_at_tpr95.min()
    else:
        #print('Interpolated')
        fpr_at_tpr95 = np.interp(0.95, df_pr['TPR'], df_pr['FPR'])
        tnr_at_tpr95 = 1 - fpr_at_tpr95
    """
    fpr_at_tpr95 =df_pr[round(df_pr['TPR'], 2) >= 0.95].min()['FPR']
    tnr_at_tpr95 = 1 - fpr_at_tpr95

    return auroc, tnr_at_tpr95, aupr

