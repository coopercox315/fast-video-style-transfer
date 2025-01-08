import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Decoder network which reconstructs the image from the style-adjusted feature maps.
    This network is a mirror like structure of the encoder network's early layers, but simplified.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            #no activation on final layer as we want raw RGB values
        )
    
    def forward(self, x):
        '''
        The 'x' tensor here is a feature map (e.g. from the encoder network).
        This network upsamples and reduces the number of channels back to 3 (RGB image).
        '''
        return self.layers(x)