import os
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100,SVHN, STL10, ImageFolder
from torchvision.transforms import transforms
from util import TwoCropTransform
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

__all__ = ['cifar10DataLoader', 'ImageNetDataLoader', 'cifar100DataLoader']

def get_transform(contrastive=True, train=True, image_size=224, dataset='cifar10', network='vit', albumentations=False, no_train_aug=False,deit=False):
    scale = (0.2, 1.)
    if network=='vit':
        if deit:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean =  (0.5, 0.5, 0.5)
            std= (0.5, 0.5, 0.5)
    else:
        if dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif dataset == 'stl10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif dataset == 'cifar100':
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        elif 'ImageNet' in dataset :
            mean =(0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            scale = (0.08, 1.0) #defaul for random resize corp
        else:
            raise ValueError('dataset not supported: {}'.format(dataset))
    if train and not no_train_aug:
        if contrastive:
            if albumentations:
                transform = TwoCropTransform(A.Compose([
                    A.RandomResizedCrop(width=image_size, height=image_size,scale=scale),
                    A.HorizontalFlip(p=0.5),
                    A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.ChannelDropout(),
                    A.Blur(p=0.5),
                    A.ColorJitter(0.4, 0.4, 0.4, 0.1),
                    A.ShiftScaleRotate(rotate_limit=90,p=0.5),
                    A.ElasticTransform(p=0.5),
                    A.InvertImg(p=0.5),],p=0.7),
                    A.ToGray(p=0.4),
                    #A.RandomGridShuffle(grid=(16,16),p=0.2),
                    A.CoarseDropout(max_height=16,max_width=16,p=0.2 if  network=='vit' else 0.0),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]),albumentations=True)
            else:
                transform = TwoCropTransform(transforms.Compose([
                    transforms.RandomResizedCrop(size=image_size, scale=scale),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomAffine(30),
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    transforms.RandomErasing(p=0.2)
                ]))
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                #transforms.RandomErasing(p=0.2)
            ])

    else:
        transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return transform

class cifar10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8, contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        if split == 'train':
            train = True
            transform = get_transform(contrastive,train=train,image_size=image_size,dataset=in_dataset,albumentations=albumentation,network=net,
                                      no_train_aug=no_train_aug, deit=deit)
        else:
            train=False
            transform = get_transform(train=train,image_size=image_size,dataset='cifar10',network=net, deit=deit)

        self.dataset = CIFAR10(root=data_dir, train=train, transform=transform, download=True)

        super(cifar10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)

class cifar100DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        if split == 'train':
            train = True
            transform = get_transform(contrastive,train=train,image_size=image_size,dataset=in_dataset,albumentations=albumentation,network=net,
                                      no_train_aug=no_train_aug, deit=deit)
        else:
            train=False
            transform = get_transform(train=train,image_size=image_size,dataset=in_dataset,network=net, deit=deit)

        self.dataset = CIFAR100(root=data_dir, train=train, transform=transform, download=True)

        super(cifar100DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)

class stl10DataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=96, batch_size=16, num_workers=8, contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        if split == 'train':
            train = True
            transform = get_transform(contrastive,train=train,image_size=image_size,dataset=in_dataset,albumentations=albumentation,network=net,
                                      no_train_aug=no_train_aug, deit=deit)
        else:
            train=False
            transform = get_transform(train=train,image_size=image_size,dataset=in_dataset,network=net, deit=deit)

        self.dataset = STL10(root=data_dir, split=split if split=='train' else 'test', transform=transform, download=True)

        super(stl10DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False if not train else True,
            num_workers=num_workers)


class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, split='train', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        if split == 'train':
            train = True
            transform = get_transform(contrastive, train=train, image_size=image_size, dataset=in_dataset,
                                      albumentations=albumentation,network=net,no_train_aug=no_train_aug, deit=deit)
        else:
            train = False
            transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        if split=='val':
            valid_index = [ind for ind, img in enumerate(self.dataset.samples) if '._1.JPEG' not in img[0]]
            self.dataset.targets = [self.dataset.targets[ind] for ind in valid_index]
            self.dataset.samples = [self.dataset.samples[ind] for ind in valid_index]

        super(ImageNetDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True if split == 'train' else False,
            num_workers=num_workers)

class CUBDataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False,in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'CUB'), transform=transform)

        super(CUBDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

class Caltech256DataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'caltech256'), transform=transform)# Caltech256(root=data_dir, transform=transform, download=True)
        super(Caltech256DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
class DogsDataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'Dogs'), transform=transform)# Caltech256(root=data_dir, transform=transform, download=True)
        super(DogsDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
class DtdDataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'Dtd'), transform=transform)# Caltech256(root=data_dir, transform=transform, download=True)
        super(DtdDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
class Places365DataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'Places365'), transform=transform)# Caltech256(root=data_dir, transform=transform, download=True)
        super(Places365DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
class FoodDataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'Food'), transform=transform)# Caltech256(root=data_dir, transform=transform, download=True)
        super(FoodDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
class Imagenet_resizeDataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'Imagenet_resize'), transform=transform)# Caltech256(root=data_dir, transform=transform, download=True)
        super(Imagenet_resizeDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
class LSUN_resizeDataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = ImageFolder(root=os.path.join(data_dir, 'LSUN_resize'), transform=transform)
        super(LSUN_resizeDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)
class SVHNDataLoader(DataLoader):
    def __init__(self, data_dir, split='val', image_size=224, batch_size=16, num_workers=8,contrastive=False, albumentation=True, net='vit',no_train_aug=False, in_dataset=None,deit=False):
        train = False
        transform = get_transform(train=train, image_size=image_size, dataset=in_dataset,network=net, deit=deit)

        self.dataset = SVHN(root=os.path.join(data_dir, 'SVHN'), transform=transform, download=True)
        super(SVHNDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)



def create_dataloaders(config,no_train_aug=False, out_dataset=False):
    dataset =  config.out_dataset if out_dataset else config.dataset
    if config.model!='vit':
        config.deit=False
    elif no_train_aug:
        config.deit = 'deit' in config.ckpt.rsplit('/',3)[1].lower()
    else:
        config.deit ='deit' in config.exp_name.lower() or 'deit' in os.path.basename(config.checkpoint_path) #
    # create dataloader
    print("Creating dataloaders for {0} with network {1} and ablumentations {2}".format(dataset, config.model, config.albumentation))
    train_dataloader = eval("{}DataLoader".format(dataset))(
        data_dir=config.out_data_dir if out_dataset else config.data_dir, # os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split='train',
        contrastive=config.contrastive,
        albumentation=config.albumentation,
        net=config.model,
        no_train_aug=no_train_aug,
        in_dataset= config.dataset,
        deit = config.deit) # for resnet where mean is different for each in dataset
    valid_dataloader = eval("{}DataLoader".format(dataset))(
        data_dir=config.out_data_dir if out_dataset else config.data_dir,  # os.path.join(config.data_dir, config.dataset),
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split='val',
        net=config.model,
        in_dataset=config.dataset,
        deit=config.deit)# for resnet where mean is different for each in dataset
    return train_dataloader, valid_dataloader

if __name__ == '__main__':
    data_loader = ImageNetDataLoader(
        data_dir='/home/hchen/Projects/vat_contrast/data/ImageNet',
        split='val',
        image_size=384,
        batch_size=16,
        num_workers=0)

    for images, targets in data_loader:
        print(targets)
