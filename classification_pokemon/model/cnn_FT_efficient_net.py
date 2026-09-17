import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EfficientNetDevelopper(nn.Module):
    def __init__(self,
        num_classes : int = 2,
        hidden_layer1: int = 1280,
        hidden_layer2: int =  256,
        dropout: float = 0.45
    ):
        super().__init__()

        #Charger EfficientNet
        self.base=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        #Renvoie les mêmes entrées de classification
        self.base.classifier = nn.Identity()

        #normalise les vecteurs 1280
        self.bn=nn.BatchNorm1d(hidden_layer1)

        # Couche dense entrée 1250 -> sortie 256
        self.fc1=nn.Linear(hidden_layer1, hidden_layer2)

        #coupe 45% des neurones pendant l'entraînement
        self.dropout=nn.Dropout(dropout)

        # Sortie 2 classes
        self.fc2=nn.Linear(hidden_layer2, num_classes)
    
    def forward(self, x):
        x=self.base(x)
        x=self.bn(x)
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)
        return x