import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import random
import numpy as np

class ToyNet(nn.Module):

    def __init__(self, head):
        super(ToyNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=2, out_features=32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(inplace=True)
        )

        if head == 'MainCE':
            self.head = nn.Linear(128, 1)
        elif head == 'SupCon' or head =='SimCLR':
            self.head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))


    def forward(self, x):
        main_feats = self.feature_extractor(x)
        head_feats = self.head(main_feats)
        return main_feats, head_feats


#############################################################
class AddGaussianNoise(object):
    """Add Gaussian Noise as transformation"""
    def __init__(self, mu=0., std=1.):
        self.mu = mu
        self.std = std

    def __call__(self, x):
        return np.asarray((x + np.random.randn(x.size)*self.std + self.mu), dtype=np.float32)

class TwoDataTransform:
    #fixme ad very small guassian noise
    """Create two close data sample from one data point"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

#############################################################

def get_random_point_data(num_samples, range_values, ood=False):

    # it gives a sequentioal array, datqset shufle will serve the randomness
    first_x_sample = [random.uniform(range_values[0][0], range_values[0][1]) for _ in range(num_samples // 2)]
    second_x_sample = [random.uniform(range_values[1][0], range_values[1][1]) for _ in range(num_samples // 2)]

    #use the same 1st range, as samples sud be seperable based on x alone
    y_sample = [random.uniform(range_values[0][0], range_values[0][1]) for _ in range(num_samples)]

    labels = np.zeros(num_samples)
    labels[num_samples//2:] = 1

    if ood:  #for ood only y is different, last samples are ood, and their labels is 2
        ood_samples = int(((num_samples/2)*0.2))
        in_out_idx = labels

        ood_y_sample = [random.uniform(range_values[1][0], range_values[1][1]) for _ in range(ood_samples)]
        second_x_sample[len(second_x_sample)-ood_samples:] = ood_y_sample
        in_out_idx[len(labels)-ood_samples:] = 2

    return np.asarray((np.vstack((np.concatenate((first_x_sample, second_x_sample)), y_sample)).T),dtype=np.float32), labels, in_out_idx if ood else None



class ToyDataset(Dataset):
    def __init__(self, samples=10000, ranges= [[1, 15],[21,35]], ood=False):

        train_transform = transforms.Compose([
            transforms.RandomApply([
                AddGaussianNoise(0., 1.)], p=0.8),
            #transforms.ToTensor(),
        ])
        self.toy_transform = TwoDataTransform(train_transform)
        self.samples, self.labels, self.in_out_labels = get_random_point_data(samples, ranges, ood)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        label = self.labels[idx]
        if self.in_out_labels is not None:
            in_out_lbl = self.in_out_labels[idx]
            return data, label, in_out_lbl
        else:
            transform_data = self.toy_transform(data)
            return transform_data, label