import argparse
from vit.src.utils import process_config


def get_eval_config():
    parser = argparse.ArgumentParser("Visual Transformer Evaluation")

    # basic config
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['t16','s16','b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=384, help="input image size", choices=[224, 384])
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    config = parser.parse_args()

    # model config
    config = eval("get_{}_config".format(config.model_arch))(config)

    print_config(config)
    return config


def get_train_config():
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic config
    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--tensorboard", default=False, action='store_true', help='flag of turnning on tensorboard')
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['t16','vs16','s16', 'b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=384, help="input image size", choices=[128, 160, 224, 384])
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--train-steps", type=int, default=10000, help="number of training/fine-tunning steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--cosine', action='store_true',help='using cosine annealing')
    parser.add_argument("--wd", type=float, default=1e-4, help='weight decay')
    parser.add_argument("--warmup-steps", type=int, default=500, help='learning rate warm up steps')
    parser.add_argument("--data-dir", type=str, default='../data', help='data folder')
    parser.add_argument("--dataset", type=str, default='ImageNet', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str,
                        help="ViT pre-trained model type")
    parser.add_argument('--eval', action='store_true',help='evaluate on dataset')
    parser.add_argument('--opt', default='SGD', type=str, choices=('AdamW', 'SGD'))
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')

    # * Mixup params
    parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0)') # later we can try it wd >0
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=0.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.0,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # similarity metric
    parser.add_argument('--sim_metric', type=str, default='Cosine', choices=['Cosine', 'Euclidean', 'Mahalanobis'],
                        help='similarity metric used in contrastive loss')
    parser.add_argument('--method', type=str, default='SupCon',  help='for SupCon,SimCLR etc')
    parser.add_argument('--head', type=str, default=None, help='for contrastive head and linear head')
    # temperature
    parser.add_argument('--temp', type=float, default=0.1,help='temperature for loss function')

    # other setting
    parser.add_argument('--contrastive', action='store_true', help='using distributed loss calculations across multiple GPUs')
    parser.add_argument('--albumentation', action='store_true', help='use albumentation as data aug')
    parser.add_argument('--test', action='store_true',
                        help='just to block the GPUs')
    parser.add_argument('--test_contrastive_acc', action='store_true',
                        help='test iterative accuracy accross all epoch for contrastive loss')
    config = parser.parse_args()

    # model config
    config = eval("get_{}_config".format(config.model_arch))(config)
    process_config(config)
    print_config(config)
    return config

def get_t16_config(config):
    """ ViT-Deit Tiny/16 configuration """
    config.patch_size = 16
    config.emb_dim = 192
    config.mlp_dim = 768
    config.num_heads = 3
    config.num_layers = 12
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config

def get_vs16_config(config):
    """ ViT-Deit Tiny/16 configuration """
    config.patch_size = 16
    config.emb_dim = 768
    config.mlp_dim = 2304
    config.num_heads = 8
    config.num_layers = 8
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config

def get_s16_config(config):
    """ ViT-Deit Small/16 configuration """
    config.patch_size = 16
    config.emb_dim = 384
    config.mlp_dim = 1536
    config.num_heads = 6
    config.num_layers = 12
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config


def get_b16_config(config):
    """ ViT-B/16 configuration """
    config.patch_size = 16
    config.emb_dim = 768
    config.mlp_dim = 3072
    config.num_heads = 12
    config.num_layers = 12
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config


def get_b32_config(config):
    """ ViT-B/32 configuration """
    config = get_b16_config(config)
    config.patch_size = 32
    return config


def get_l16_config(config):
    """ ViT-L/16 configuration """
    config.patch_size = 16
    config.emb_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config


def get_l32_config(config):
    """ Vit-L/32 configuration """
    config = get_l16_config(config)
    config.patch_size = 32
    return config


def get_h14_config(config):
    """  ViT-H/14 configuration """
    config.patch_size = 14
    config.emb_dim = 1280
    config.mlp_dim = 5120
    config.num_heads = 16
    config.num_layers = 32
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config


def print_config(config):
    message = ''
    message += '----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)