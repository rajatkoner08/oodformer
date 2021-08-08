from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import sklearn.covariance
from scipy.spatial.distance import pdist, cdist, squareform
from util import  accuracy



# lid of a batch of query points X
def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg, pos_in, pos_out, mean_in, mean_out):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_pos_all = np.concatenate((X_pos, pos_in, mean_in), axis=1)

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X_neg_all = np.concatenate((X_neg, pos_out, mean_out), axis=1)

    X = np.concatenate((X_pos_all, X_neg_all))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y


def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=True)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        output, out_features = model.call_feature_list(data)
        #class_dist = classifier(output)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
        """
        # compute the accuracy
        #acc1, acc5 = accuracy(class_dist, target.cuda(), topk=(1, 5))
        pred = class_dist.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()"""

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                        = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    covariance = []
    for k in range(num_output):
        X = 0
        class_precision = []
        class_covariance = []
        for i in range(num_classes):
            """
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)"""
            X_ = list_features[k][i] - sample_class_mean[k][i]
            group_lasso.fit(X_.cpu().numpy())
            temp_precision = group_lasso.precision_
            temp_precision = torch.from_numpy(temp_precision).float().cuda()
            class_precision.append(temp_precision)
            temp_covariance = group_lasso.covariance_
            temp_covariance = torch.from_numpy(temp_covariance).float().cuda()
            class_covariance.append(temp_covariance)


        # find inverse
        """
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        temp_covariance = group_lasso.covariance_
        temp_covariance = torch.from_numpy(temp_covariance).float().cuda()"""
        precision.append(class_precision)
        covariance.append(class_covariance)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision, covariance


def get_Mahalanobis_score(model, test_loader, num_classes, outf, out_flag, net_type, sample_mean, precision,
                            covariance, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    #fixme add model output into classifier to get the class dist
    ## fixed - added classifier, but not used- check again the usage of classifier

    Mahalanobis = []
    position = []
    closest_class_mean = []

    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt' % (outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt' % (outf, str(layer_index))

    g = open(temp_file_name, 'w')

    for test_data, target in test_loader:

        test_data, target = test_data.cuda(), target.cuda()
        test_data.requires_grad = True

        out_features = model.intermediate_forward(test_data, layer_index)
        out_feat_shape = out_features.shape
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        num_feat = out_feat_shape[1]

        # compute Mahalanobis score
        gaussian_score = 0

        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            #term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()

            """
            print((torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t())).shape) # ([10, 10])
            print((torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()).shape) # ([10])
            print((covariance[layer_index]).shape) # ([64, 64])
            print(((covariance[layer_index]).diag()).shape) # ([64])
            print((torch.det(covariance[layer_index])).shape)
            print(torch.det(covariance[layer_index])) # tensor(0., device='cuda:0')
            print(torch.pow((torch.tensor(2*np.pi).cuda()), num_feat)) # tensor(inf, device='cuda:0')
            """
            cov_mat = covariance[layer_index][i]
            epsilon = 0.00001
            term_gau = - torch.mm(torch.mm(zero_f, precision[layer_index][i]), zero_f.t()).diag()
            term_norm = - torch.tensor(num_feat * np.log(2*np.pi) + np.linalg.slogdet(cov_mat.cpu().numpy() + epsilon * np.identity(cov_mat.shape[0]))[1], dtype=torch.float64) # Additional term based on Contrastive Training paper
            total_score = term_gau + term_norm
            if i == 0:
                gaussian_score = total_score.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, total_score.view(-1, 1)), 1)
        """
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        #gradient = torch.ge(data.grad.data, 0)
        gradient = torch.ge(test_data.grad.data, 0) ## check for None type data problem
        gradient = (gradient.float() - 0.5) * 2
        if 'densenet' in net_type:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        elif 'resnet' in net_type:
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        tempInputs = torch.add(test_data.data, -magnitude, gradient)

        noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
       
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            #zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
        """

        max_gaussian_score, pos = torch.max(gaussian_score, dim=1)
        mean_for_max_score = []
        for p in pos:
            m = sample_mean[layer_index][p]
            m_= torch.mean(m)
            mean_for_max_score.append(m_)
        closest_class_mean.extend(mean_for_max_score)
        position.extend(pos.cpu().numpy())
        Mahalanobis.extend(max_gaussian_score.cpu().numpy())


        for i in range(test_data.size(0)):
            g.write("{}\n".format(max_gaussian_score[i]))
    g.close()

    return Mahalanobis, position, closest_class_mean  #, out_features_full if out_flag else None


def get_posterior(model, classifier, net_type, test_loader, magnitude, temperature, outf, out_flag):
    '''
    Compute the maximum value of (processed) posterior distribution - ODIN
    return: null
    '''
    #fixme add model output into classifier to get the class dist
    criterion = nn.CrossEntropyLoss()
    total = 0
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt' % (outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt' % (outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt' % (outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt' % (outf)

    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')

    for data, _ in test_loader:
        total += data.size(0)
        data = data.cuda()
        batch_output = model(data)

        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        tempInputs = torch.add(data.data, -magnitude, gradient)
        outputs = model(tempInputs)
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)

        for i in range(data.size(0)):
            if total <= 1000:
                g.write("{}\n".format(soft_out[i]))
            else:
                f.write("{}\n".format(soft_out[i]))

    f.close()
    g.close()


def get_Mahalanobis_score_adv(model, classifier, test_data, test_label, num_classes, outf, net_type, sample_mean, precision,
                              layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    return: Mahalanobis score from layer_index
    '''
    #fixme add model output into classifier to get the class dist
    Mahalanobis = []
    batch_size = 100
    total = 0

    for data_index in range(int(np.floor(test_data.size(0) / batch_size))):
        target = test_label[total: total + batch_size].cuda()
        data = test_data[total: total + batch_size].cuda()
        total += batch_size

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1 / 255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7 / 255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
                                 gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        tempInputs = torch.add(data.data, -magnitude, gradient)

        noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1, 1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())

    return Mahalanobis


def get_LID(model, test_clean_data, test_adv_data, test_noisy_data, test_label, num_output):
    '''
    Compute LID score on adversarial samples
    return: LID score
    '''
    #fixme add model output into classifier to get the class dist
    total = 0
    batch_size = 100

    LID, LID_adv, LID_noisy = [], [], []
    overlap_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    for i in overlap_list:
        LID.append([])
        LID_adv.append([])
        LID_noisy.append([])

    for data_index in range(int(np.floor(test_clean_data.size(0) / batch_size))):
        data = test_clean_data[total: total + batch_size].cuda()
        adv_data = test_adv_data[total: total + batch_size].cuda()
        noisy_data = test_noisy_data[total: total + batch_size].cuda()
        target = test_label[total: total + batch_size].cuda()

        total += batch_size

        output, out_features = model.feature_list(data)
        X_act = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))

        output, out_features = model.feature_list(adv_data)
        X_act_adv = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act_adv.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))

        output, out_features = model.feature_list(noisy_data)
        X_act_noisy = []
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            X_act_noisy.append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].size(0), -1)))

        # LID
        list_counter = 0
        for overlap in overlap_list:
            LID_list = []
            LID_adv_list = []
            LID_noisy_list = []

            for j in range(num_output):
                lid_score = mle_batch(X_act[j], X_act[j], k=overlap)
                lid_score = lid_score.reshape((lid_score.shape[0], -1))
                lid_adv_score = mle_batch(X_act[j], X_act_adv[j], k=overlap)
                lid_adv_score = lid_adv_score.reshape((lid_adv_score.shape[0], -1))
                lid_noisy_score = mle_batch(X_act[j], X_act_noisy[j], k=overlap)
                lid_noisy_score = lid_noisy_score.reshape((lid_noisy_score.shape[0], -1))

                LID_list.append(lid_score)
                LID_adv_list.append(lid_adv_score)
                LID_noisy_list.append(lid_noisy_score)

            LID_concat = LID_list[0]
            LID_adv_concat = LID_adv_list[0]
            LID_noisy_concat = LID_noisy_list[0]

            for i in range(1, num_output):
                LID_concat = np.concatenate((LID_concat, LID_list[i]), axis=1)
                LID_adv_concat = np.concatenate((LID_adv_concat, LID_adv_list[i]), axis=1)
                LID_noisy_concat = np.concatenate((LID_noisy_concat, LID_noisy_list[i]), axis=1)

            LID[list_counter].extend(LID_concat)
            LID_adv[list_counter].extend(LID_adv_concat)
            LID_noisy[list_counter].extend(LID_noisy_concat)
            list_counter += 1

    return LID, LID_adv, LID_noisy
