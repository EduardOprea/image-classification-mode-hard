import argparse
import copy
import pandas as pd
from torchvision import transforms
from dataset import Dataset
import torchvision.models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.optim
import time
from tqdm import tqdm
import pickle


from run_metadata import RunMetadata

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
    
def get_preprocess_transforms(model_name: str):
    if model_name == 'resnet50':
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 
        return weights.transforms()


def train_model(model, device, dataloaders, criterion, optimizer, run_args: RunMetadata):
    since = time.time()

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
                val_acc_history.append(epoch_acc)

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


def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training an image classifier')

    parser.add_argument('--model_name', type=str, default="resnet50",
                        help="Name of convnet model")

    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16)')

    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs (default: 40)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (SGD) (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (SGD) (default: 0.9)')

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


    parsed_arguments = parser.parse_args()

    return parsed_arguments

if __name__ == '__main__':
    args = parse_command_line_arguments()

    for k, v in args.__dict__.items():
        print(k + '=' + str(v))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    print("Training on ", device)
    df = pd.read_csv(args.annotations_file)
    num_classes = len(set(df['label']))

    run_args = RunMetadata(args.model_name, num_classes, args.epochs, args.freq_ckpt, args.results_dir)

    # feature_extract = False means we will update all the parameter, not only the last linear layer
    model = init_model(num_classes, args.model_name, feature_extract = args.feature_extract)
    preprocess = get_preprocess_transforms(args.model_name)
    model = model.to(device)


    dataset = Dataset(args.rootdir, df, preprocess)
    [train_set, val_set] = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_set, shuffle = True, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, shuffle = True, batch_size=args.batch_size)

    dataloaders = {'train': train_loader, 'val': val_loader }

    optimizer = get_optimizer(model, feature_extract=args.feature_extract, lr = args.lr, momentum= args.momentum)

    criterion = nn.CrossEntropyLoss()
    train_model(model, device, dataloaders, criterion, optimizer, run_args)