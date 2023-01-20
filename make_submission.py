import argparse
import torchvision
import torch
from torch.utils.data import DataLoader
from dataset import UnlabeledDataset
from tqdm import tqdm
from torch import nn
import pandas as pd
from models.ensemble_model import EnsembleModel

def parse_command_line_arguments():

    parser = argparse.ArgumentParser(
        description='CLI for training an image classifier')

    parser.add_argument('--model_name', type=str, default="ensemble",
                        help="The arch to use")
    parser.add_argument('--checkpoint_path', type=str, default="results/checkpoints/t1_ensemble_pretrained_models_smooth_label_v2.pth",
                        help="Path to model weights")
    parser.add_argument('--batch_size', type=int, default=16,
                help="Batch size")

    parser.add_argument('--rootdir', type=str, default="E:\Fisierele mele\Facultate\AAIT\HW2\\task1\\val_data",
                        help="Path to the images folder")

    parser.add_argument('--output_file', type=str, default="submission.csv",
                        help="Name of submission file")


    parsed_arguments = parser.parse_args()
    return parsed_arguments

def load_model_from_ckpt(model, path, num_classes):
    if model == 'resnet50':
        model = torchvision.models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(path))
        return model
    elif model == 'resnet152':
        model = torchvision.models.resnet152()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(path))
        return model
    elif model == 'ensemble':
        print("Loading ensemble model")
        resnet_model = torchvision.models.resnet152()
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs,num_classes)
        
        densenet_model = torchvision.models.densenet161()
        num_ftrs = densenet_model.classifier.in_features
        densenet_model.classifier = nn.Linear(num_ftrs, num_classes)

        vgg_model = torchvision.models.vgg19_bn()
        num_ftrs = vgg_model.classifier[6].in_features
        vgg_model.classifier[6] = nn.Linear(num_ftrs,num_classes)   
        model = EnsembleModel(resnet_model, densenet_model, vgg_model, num_classes)
        model.load_state_dict(torch.load(path))
        model.freeze_ensemble_models_params()
        model.freeze_classifier_params
        return model

def init_resnet50_model(num_classes):
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs,num_classes)
    return model
if __name__ == '__main__':
    args = parse_command_line_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2 
    dataset = UnlabeledDataset(args.rootdir, weights.transforms())
    dataloader = DataLoader(dataset, args.batch_size)
    
    #model = init_resnet50_model(100)
    model = load_model_from_ckpt(args.model_name, args.checkpoint_path, num_classes=100)
    model.to(device)
    model.eval()
    
    images = []
    predictions = []
    
    
    
    with torch.no_grad():
        for inputs, images_names in tqdm(dataloader):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                images += images_names
                predictions += preds.tolist()
                

    submission_dict = {'sample': images, 'label': predictions} 
    submission_df = pd.DataFrame(submission_dict)
    submission_df.to_csv(args.output_file, index=False)
    



