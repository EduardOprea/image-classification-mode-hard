import torch.nn as nn
import torchvision.models as models

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, use_pretrained = True):
        super(ResNetSimCLR, self).__init__()
        if use_pretrained == True:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else: 
            weights = None
            
        self.resnet_dict = {"resnet50": models.resnet50(weights)}


        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise KeyError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
