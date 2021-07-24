import timm
from torch import nn


class BrainTumor2dModel(nn.Module):
    def __init__(self, model_arch, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.model(x)
        return out
