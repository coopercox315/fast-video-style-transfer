import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    """
    Encoder network using VGG19. Outputs content and style features at desired layers.
    """
    def __init__(self, layers=(0, 5, 10, 19, 28), requires_grad=False):
        super(Encoder, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        self.selected_layers = layers #layers to extract features from
        self.layers = nn.ModuleList() #list of layers to extract features from
        for idx, layer in enumerate(vgg):
            self.layers.append(layer)

        if not requires_grad: #freeze layers if not training
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        '''
        Forward pass:
        - Go through each layer in the VGG model
        - Collect features from the selected layers
        - Return the features in a dictionary
        '''
        features = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.selected_layers:
                features[i].append(x)
        return features
