import argparse
import copy
import os
import pandas as pd
from torchvision import transforms
from dataset import LabeledDataset
import torchvision.models
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim
import time
from tqdm import tqdm
import pickle
from models.ensemble_model import EnsembleModel
from models.resnet_simclr import ResNetSimCLR
import json
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


from run_metadata import RunMetadata
from train_noise_correction import AverageMeter, accuracy

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def init_model(num_classes, model_name: str, feature_extract = False):
    if model_name == 'resnet50':
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        return model
    elif model_name == 'resnet152':
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet152(weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        return model
    elif model_name == 'densenet161':
        weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1
        model = torchvision.models.densenet161(weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        return model
    elif model_name == 'vgg19':
        weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        model = torchvision.models.vgg19_bn(weights)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)   
        return model
     
def get_preprocess_transforms(model_name: str):
    if model_name == 'resnet50':
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 
        return weights.transforms()
    if model_name == 'resnet152':
        weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2 
        return weights.transforms()
    if model_name == 'densenet161':
        weights = torchvision.models.DenseNet161_Weights.IMAGENET1K_V1 
        return weights.transforms()
    if model_name == 'vgg19':
        weights = torchvision.models.VGG19_BN_Weights.IMAGENET1K_V1
        return weights.transforms()

def loss_gls(logits, labels, smooth_rate=0.1):
    # logits: model prediction logits before the soft-max, with size [batch_size, classes]
    # labels: the (noisy) labels for evaluation, with size [batch_size]
    # smooth_rate: could go either positive or negative, 
    # smooth_rate candidates we adopted in the paper: [0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -4.0, -6.0, -8.0].
    confidence = 1. - smooth_rate
    logprobs = F.log_softmax(logits, dim=-1)
    nll_loss = -logprobs.gather(dim=-1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss = confidence * nll_loss + smooth_rate * smooth_loss
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    return torch.sum(loss)/num_batch

def train_model(model, device, dataloaders, criterion, optimizer, run_args: RunMetadata):
    since = time.time()
    tb_logdir = os.path.join(run_args.results_dir,"tb_logs")
    tb_writer = SummaryWriter(tb_logdir)
    print("Saving the tensorboard run logs in ",tb_logdir)

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(run_args.num_epochs):
        print('Epoch {}/{}'.format(epoch, run_args.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'{run_args.results_dir}/checkpoints/bestacc.pth')
            if phase == 'val':
                tb_writer.add_scalar('loss_val', epoch_loss, global_step=epoch)
                tb_writer.add_scalar('acc_val', epoch_acc, global_step=epoch)
                val_acc_history.append(epoch_acc)
            elif phase == 'train':
                tb_writer.add_scalar('loss_train', epoch_loss, global_step=epoch)
                tb_writer.add_scalar('acc_train', epoch_acc, global_step=epoch)


        if (epoch + 1) % run_args.freq_ckpt == 0:
            torch.save(model.state_dict(), f'{run_args.results_dir}/checkpoints/epoch{epoch}.pth')

        print()


    torch.save(model.state_dict(), f'{run_args.results_dir}/checkpoints/epoch{epoch}.pth')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    pickle_list(val_acc_history, f'{run_args.results_dir}/acc_evolution.pkl')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def pickle_list(list, file):
    with open(file,"wb") as fp:
        pickle.dump(list, fp)

def get_optimizer(model, feature_extract, lr = 0.001, momentum = 0.9):
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum)
    return optimizer 

def get_ensemble_transforms(input_size = 224):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])}
    # use the same transforms for train and validation because the way the dataset is setup
    return data_transforms['train']

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training an image classifier')

    parser.add_argument('--model_name', type=str, default="densenet161",
                        help="Name of convnet model")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs (default: 40)')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (SGD) (default: 1e-4)')
    parser.add_argument('--lr2', type=float, default=0.2,
                        help='learning rate (SGD) (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (SGD) (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
    

    parser.add_argument('--alpha', default=0.4, type=float,
                    metavar='H-P', help='the coefficient of Compatibility Loss')
    parser.add_argument('--beta', default=0.1, type=float,
                        metavar='H-P', help='the coefficient of Entropy Loss')
    parser.add_argument('--lambda1', default=600, type=int,
                        metavar='H-P', help='the value of lambda')


    parser.add_argument('--stage1', default=3, type=int,
                        metavar='H-P', help='number of epochs utill stage1')
    parser.add_argument('--stage2', default=6, type=int,
                        metavar='H-P', help='number of epochs utill stage2')

    parser.add_argument('--device', default='cpu', type=str,
                        help='device to be used for computations (in {cpu, cuda:0, cuda:1, ...}, default: cpu)')

    parser.add_argument('--freq_ckpt', type=int, default=5,
                        help='Frequency of checkpointing')

    parser.add_argument('--results_dir', type=str, default="results",
                        help='Directory where results are logged')
    parser.add_argument('--rootdir', type=str, required=True,
                        help='Root directory for the tasks')

    parser.add_argument('--annotations_file', type=str, required=True,
                        help='Path to annotations.csv')

    parser.add_argument('--feature_extract', type=bool, default= False,
                        help='The conv net is used just as a feature extractor')
    
    parser.add_argument('--use_label_smoothing', action='store_true',
                        help='If the loss is calculated with smoothed labels')
    


    parser.add_argument('--use_noise_correction', action='store_true',
                        help='If this is true, It will apply noise correction')
    


    parser.add_argument('--use_ensemble', action='store_true',
                        help='If this is true, it will use an asemble of models : resnet152, vgg19 and densenet161')
    
    parser.add_argument('--freeze_ensemble_models', action='store_true',
                        help='If set, it will freeze the parameters of the ensemble models, and train only the last ensembling linear layer')
    parser.add_argument('--use_checkpoints_ensemble', action='store_true',
                        help='If set, it will use checkpoint weights for the ensemble model')
    parser.add_argument('--resnet_ckpt', type=str,
                        help='Path to ensemble resnet checkpoint')
    parser.add_argument('--densenet_ckpt', type=str,
                        help='Path to ensemble densenet checkpoint')
    parser.add_argument('--vgg_ckpt', type=str,
                        help='Path to ensemble vgg checkpoint')
    
    
    parser.add_argument('--smooth_rate', type=float, default=0.0,
                        help='The smooth rate of label smoothing')


    parser.add_argument('--use_custom_checkpoint', type=bool, default= False,
                        help='The weights are loaded from a custom path ( this will only work for resnet50 )')

    parser.add_argument('--custom_ckpt_path', type=str, default= "results/checkpoints/checkpoint_0200.pth.tar",
                        help='Path to the custom weights')
    
    parser.add_argument('--train_split_percentage', type=float, default= 0.9,
                        help="Percentage of images to be used for train split")
    
    
    parser.add_argument('--val_split_percentage', type=float, default= 0.1,
                        help="Percentage of images to be used for val split")


    parsed_arguments = parser.parse_args()

    return parsed_arguments

def get_run_metadata_obj(args) -> RunMetadata:
    run_args = RunMetadata(model = args.model_name,output_size= num_classes,
         num_epochs= args.epochs, batch_size = args.batch_size,
         freq_ckpt= args.freq_ckpt, results_dir= args.results_dir,
         lr = args.lr, momentum= args.momentum, optimizer='adam', 
         use_label_smoothing=args.use_label_smoothing, smooth_rate=args.smooth_rate, use_ensemble=args.use_ensemble,
         freeze_ensemble_models = args.freeze_ensemble_models)
    run_args.lr2 = args.lr2
    run_args.alpha = args.alpha
    run_args.beta = args.beta
    run_args.stage1 = args.stage1
    run_args.stage2 = args.stage2
    run_args.lambda1 = args.lambda1
    return run_args

def load_model_from_ckpt(num_classes, path, feature_extract):
    model = torchvision.models.resnet50()
    checkpoint_state = torch.load(path)
    simclrModel = ResNetSimCLR("resnet50", num_classes, use_pretrained=False)
    simclrModel.load_state_dict(checkpoint_state['state_dict'])
    # strict = False because they will not match in the last layer which will be dropped anyway
    model.load_state_dict(simclrModel.backbone.state_dict(), strict=False)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,num_classes)
    return model
def load_ensemble_model(num_classes, args):
    if args.use_checkpoints_ensemble == False:
        resnet_model = init_model(num_classes, "resnet152", args.feature_extract)
        densenet_model = init_model(num_classes, "densenet161", args.feature_extract)
        vgg_model = init_model(num_classes, "vgg19", args.feature_extract)
        model = EnsembleModel(resnet_model, densenet_model, vgg_model, num_classes)
        if args.freeze_ensemble_models == True:
            print("Freezing parameters of ensemble models")
            model.freeze_ensemble_models_params()
        return model
    else:
        print("loading checkpoint weights for ensemble model")
        resnet_model = init_model(num_classes, "resnet152", args.feature_extract)
        resnet_model.load_state_dict(torch.load(args.resnet_ckpt))
        densenet_model = init_model(num_classes, "densenet161", args.feature_extract)
        densenet_model.load_state_dict(torch.load(args.densenet_ckpt))
        vgg_model = init_model(num_classes, "vgg19", args.feature_extract)
        vgg_model.load_state_dict(torch.load(args.vgg_ckpt))
        model = EnsembleModel(resnet_model, densenet_model, vgg_model, num_classes)
        if args.freeze_ensemble_models == True:
            print("Freezing parameters of ensemble models")
            model.freeze_ensemble_models_params()
        return model

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
def save_run_metadata(run_args, path):
    with open(path,"w") as f:
        json.dump(run_args, f, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)


def adjust_learning_rate(optimizer, epoch, run_args: RunMetadata):
    """Sets the learning rate"""
    if epoch < run_args.stage2 :
        lr = run_args.lr
    elif epoch < (run_args.epochs - run_args.stage2)//3 + run_args.stage2:
        lr = run_args.lr2
    elif epoch < 2 * (run_args.epochs - run_args.stage2)//3 + run_args.stage2:
        lr = run_args.lr2//10
    else:
        lr = run_args.lr2//100
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, index) in enumerate(val_loader):
            target = target.to(device)
            input = input.to(device)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

    return top1.avg
     
def train_noise_correction(trainloader, valloader, optimizer, y_file, run_args: RunMetadata):
    print("y file ", y_file)
    best_prec1 = 0.0
    for epoch in range(run_args.num_epochs):
        adjust_learning_rate(optimizer, epoch, run_args)
        
        if os.path.isfile(y_file):
            print("Loading y from file")
            y = np.load(y_file)
        else:
            y = []

        update_weights_noise_correction(trainloader, model, criterion, optimizer, epoch, y, run_args)
        # evaluate on validation set
        prec1 = validate(valloader, model, criterion)
        print("Precision is", prec1)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if is_best == True:
           torch.save(model.state_dict(), f'{run_args.results_dir}/checkpoints/bestacc.pth')


def update_weights_noise_correction(train_loader, model, criterion, optimizer, epoch, y, run_args: RunMetadata):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # switch to train mode
    model.train()

    end = time.time()

    # new y is y_tilde after updating
    new_y = np.zeros([len(train_loader.dataset),run_args.output_size])

    for i, (input, target, index) in enumerate(train_loader):
        # measure data loading time

        data_time.update(time.time() - end)

        index = index.numpy()

        target1 = target.to(device)
        input = input.to(device)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target1)

        # compute output
        output = model(input_var)

        logsoftmax = nn.LogSoftmax(dim=1).cuda()
        softmax = nn.Softmax(dim=1).cuda()
        if epoch < run_args.stage1:
            # lc is classification loss
            lc = criterion(output, target_var)
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(target.size(0), run_args.output_size).scatter_(1, target.view(-1, 1), 10.0)
            onehot = onehot.numpy()
            new_y[index, :] = onehot
        else:
            yy = y
            yy = yy[index,:]
            yy = torch.FloatTensor(yy)
            yy = yy.to(device)
            yy = torch.autograd.Variable(yy,requires_grad = True)
            # obtain label distributions (y_hat)
            last_y_var = softmax(yy)
            lc = torch.mean(softmax(output)*(logsoftmax(output)-torch.log((last_y_var))))
            # lo is compatibility loss
            lo = criterion(last_y_var, target_var)
        # le is entropy loss
        le = - torch.mean(torch.mul(softmax(output), logsoftmax(output)))

        if epoch < run_args.stage1:
            loss = lc
        elif epoch < run_args.stage2:
            loss = lc + run_args.alpha * lo + run_args.beta * le
        else:
            loss = lc

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target1, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch >= run_args.stage1 and epoch < run_args.stage2:
            lambda1 = run_args.lambda1
            # update y_tilde by back-propagation
            yy.data.sub_(lambda1*yy.grad.data)

            new_y[index,:] = yy.data.cpu().numpy()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
    if epoch < run_args.stage2:
        # save y_tilde
        y = new_y
        y_file = os.path.join(run_args.results_dir, "y.npy")
        print("Saving y to ", y_file)
        np.save(y_file,y)
        # y_record = run_args.results_dir + "y_%03d.npy" % epoch
        # np.save(y_record,y)

if __name__ == '__main__':
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    ensure_dir_exists(os.path.join(args.results_dir,'checkpoints'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print("Training on ", device)
    df = pd.read_csv(args.annotations_file)
    num_classes = len(set(df['label']))

    run_args = get_run_metadata_obj(args)
    save_run_metadata(run_args, os.path.join(args.results_dir,"run_metadata.json"))
    
    # feature_extract = False means we will update all the parameter, not only the last linear layer
    if args.use_custom_checkpoint == True:
       model = load_model_from_ckpt(num_classes, args.custom_ckpt_path, args.feature_extract)
       preprocess = get_preprocess_transforms(args.model_name)
    elif args.use_ensemble == True:
        model = load_ensemble_model(num_classes, args)
        #preprocess = get_ensemble_transforms(input_size = 224)
        preprocess = get_preprocess_transforms('resnet152')
    else:
        model = init_model(num_classes, args.model_name, feature_extract = args.feature_extract)
        preprocess = get_preprocess_transforms(args.model_name)
    
    
    
    model = model.to(device)


    # dataset = LabeledDataset(args.rootdir, df, preprocess)
    # [train_set, val_set] = random_split(dataset, [args.train_split_percentage, args.val_split_percentage])

    train_df, val_df = train_test_split(df, test_size = args.val_split_percentage)
    

    train_set = LabeledDataset(args.rootdir, train_df, preprocess)
    val_set = LabeledDataset(args.rootdir, val_df, preprocess)

    train_loader = DataLoader(train_set, shuffle = True, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, shuffle = True, batch_size=args.batch_size)

    dataloaders = {'train': train_loader, 'val': val_loader }

    optimizer = get_optimizer(model, feature_extract=args.feature_extract, lr = args.lr, momentum= args.momentum)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.smooth_rate)

    if args.use_noise_correction:
        # optimizer = torch.optim.SGD(model.parameters(), run_args.lr,
        #                             momentum=run_args.momentum,
        #                             weight_decay=args.weight_decay)
        train_noise_correction(train_loader, val_loader, optimizer, os.path.join(run_args.results_dir, "y.npy"), run_args)
    else:
        
        train_model(model, device, dataloaders, criterion, optimizer, run_args)