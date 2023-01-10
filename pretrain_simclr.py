import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from data_aug.gaussian_blur import GaussianBlur
from dataset import ContrastiveDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from torchvision import transforms
from data_aug.view_generator import ContrastiveLearningViewGenerator
from utils import get_file_names

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-rootdir_labeled', metavar='DIR', default='E:\Fisierele mele\Facultate\AAIT\HW2\\task1\\train_data\images\\labeled', help='path to labeled images')
parser.add_argument('-rootdir_unlabeled', metavar='DIR', default='E:\Fisierele mele\Facultate\AAIT\HW2\\task1\\train_data\images\\unlabeled', help='path to unlabeled images')
parser.add_argument('-results_dir', metavar='DIR', default='results', help='results log')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--use_pretrained_weights', action='store_true',
                    help='use pretrained weights for model')                        

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('-cf', '--checkpoint_freq', default=5, type=int,
                    help='frequency of saving checkpoints')

parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,

                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')

parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=100, type=int,
                    help='feature dimension (default: 100)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

def main():
    args = parser.parse_args()
    print("CMD arguments \n", args)
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."

    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1


    transform = ContrastiveLearningViewGenerator(get_simclr_pipeline_transform(64),
                                                                  args.n_views)
    labeled_imgs = get_file_names(args.rootdir_labeled)
    labeled_imgs = [os.path.join(args.rootdir_labeled,img) for img in labeled_imgs]
    unlabeled_imgs = get_file_names(args.rootdir_unlabeled)
    unlabeled_imgs = [os.path.join(args.rootdir_unlabeled,img) for img in unlabeled_imgs]
    all_imgs = labeled_imgs + unlabeled_imgs

    train_dataset = ContrastiveDataset(all_imgs, transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim, use_pretrained=args.use_pretrained_weights)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
