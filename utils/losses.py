import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gram_matrix(features):
    """
    Compute the Gram matrix of a given input tensor.
    Args:
    - input (Tensor): the input tensor, [b, c, h, w]
    
    Returns:
    - Tensor: the Gram matrix of the input tensor, [b, c, c] capturing style correlations.
    """
    b, c, h, w = features.size()
    
    #Reshape the input tensor to be a 2D matrix
    features = features.view(b, c, h * w) #each samples channels are seperate
    #different to gatys et al. (2016), we compute the Gram matrix for each sample in the batch as feed forward uses batch size > 1
    
    #Compute the Gram matrix
    g_mat = torch.mm(features, features.t(1,2))
    
    #return the normalized Gram matrix
    return g_mat.div(c * h * w)

class LossNetwork(nn.Module):
    """
    A small wrapper that uses a pretrained VGG19 for content/style loss.
    We freeze its params and track a few layers for style and one for content.
    """
    def __init__(self, content_layer=21, style_layers=(0, 5, 10, 19, 28)):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval() 

        self.vgg_layers = nn.ModuleList([layer for layer in vgg]) #getting the layers of VGG19

        self.content_layer_idx = content_layer 
        self.style_layer_idxs = style_layers 

        #freeze params
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        x: (B, C, H, W) image tensor
        Returns:
        - A dictionary of features {layer_idx:feature_map} we can use for content/style loss.
        """
        features = {}
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i == self.content_layer_idx:
                features['content'] = x
            if i in self.style_layer_idxs:
                features[i] = x
        
        return features
    
#functions instead of classes (like gatys), as we don't need to store state or inject loss layers into the feed-forward pipeline.
#In feed-forward, we calculate losses after generating the stylized image, so functions are more appropriate.
def content_loss(stylized_feat, content_feat):
    return F.mse_loss(stylized_feat, content_feat)

def style_loss(stylized_feat, style_feat):
    """
    Compute the style loss between the stylized features and the style features.
    Stylized features are the output of the feed-forward network, while style features are the output of the loss network.
    Stylized_feat: stylized_img = model(content_img, style_img) -> loss_network(stylized_img)
    Style_feat: loss_network(style_img)
    """
    G_stylized = gram_matrix(stylized_feat)
    G_style = gram_matrix(style_feat)

    return F.mse_loss(G_stylized, G_style)




