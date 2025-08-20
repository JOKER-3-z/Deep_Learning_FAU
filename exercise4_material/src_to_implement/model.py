import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    """
    ResNet-based classifier using a pretrained model.
    """
    def __init__(self, num_classes=2, dropout_prob=0.5, pretrained=True):
        super(ResNet, self).__init__()
        # Load Pretrained ResNet34
        self.model = models.resnet34(pretrained=pretrained)
        # Modify the final fully connected (FC) layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_prob), 
            nn.Linear(in_features, num_classes),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, x):
        return self.model(x)