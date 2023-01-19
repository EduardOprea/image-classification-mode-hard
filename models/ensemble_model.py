from torch import nn
import torch
class EnsembleModel(nn.Module):   
    def __init__(self, modelA, modelB, modelC, output_size):
        super().__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Linear(output_size * 3, output_size)
        
    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        x3 = self.modelC(x)
        x = torch.cat((x1, x2, x3), dim=1)
        out = self.classifier(x)
        return out
    
    def freeze_ensemble_models_params(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True 

    def freeze_classifier_params(self):
        for param in self.classifier.parameters():
            param.requires_grad = False         